"""``ask_parent``: a subagent parks on a decision and receives the answer in its tool result.

The interesting cases are all about the wait, not the message: the answer arriving from another
thread, the deadline winning, the child being stopped while parked, a second question while one is
outstanding, and an answer from a conversation that does not own the child. These tests drive the
rendezvous directly, so none of them needs a real spawn or a live gateway.
"""

import json
import threading
import time
import weakref

import pytest

from tools import subagent_question_tool as questions
from tools.delegate_tool import _register_subagent, _unregister_subagent
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification


class _Parent:
    def __init__(self, session_id="sess-parent"):
        self.session_id = session_id
        self._current_task_id = "parent-turn-1"


class _Child:
    def __init__(self, parent):
        self._delegate_parent_ref = weakref.ref(parent)
        self._delegate_stop_requested = False


def _child_pair(sid, goal="pick a report language"):
    """Register a child and keep the parent alive: the child holds only a weakref, as a spawn does."""
    parent = _Parent()
    child = _Child(parent)
    _register_subagent({"subagent_id": sid, "parent_id": None, "depth": 0, "goal": goal,
                        "model": "m", "started_at": time.time(), "status": "running",
                        "tool_count": 0, "agent": child,
                        "owner_agent_session_id": parent.session_id})
    return sid, parent, child


@pytest.fixture(autouse=True)
def _plain_spawn(monkeypatch):
    import tools.process_registry as _pr
    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)


@pytest.fixture(autouse=True)
def _empty_rendezvous():
    with questions._pending_lock:
        questions._pending.clear()
    yield
    with questions._pending_lock:
        questions._pending.clear()


@pytest.fixture
def clean_queue():
    def drain():
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
    drain()
    yield process_registry.completion_queue
    drain()


@pytest.fixture
def parent_session(monkeypatch):
    """A deliverable host with a known conversation identity."""
    from gateway import session_context

    env = {"HERMES_SESSION_KEY": "desktop:conv-1", "HERMES_SESSION_ID": "conv-1",
           "HERMES_SESSION_PLATFORM": "desktop"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": env.get(key, default))
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)


def _parked_call(sid, holder, **kwargs):
    """Run ask_parent in a worker thread, the way a real child's tool call is parked."""
    thread = threading.Thread(target=lambda: holder.update(result=questions.ask_parent_tool(task_id=sid, **kwargs)),
                              daemon=True)
    thread.start()
    return thread


def _await_pending(timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with questions._pending_lock:
            if questions._pending:
                return next(iter(questions._pending.values()))
        time.sleep(0.02)
    raise AssertionError("the child never registered a pending question")


def test_the_answer_unparks_the_child_and_becomes_its_tool_result(clean_queue, parent_session):
    sid, parent, _child = _child_pair("sa-0-ask0001")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="Bulgarian or English for the report?",
                              options=["Bulgarian", "English"], wait_seconds=10)
        entry = _await_pending()

        receipt = json.loads(questions.answer_pending_question(sid, entry.question_id, "English", parent))
        assert receipt["status"] == "delivered"

        thread.join(timeout=5)
        result = json.loads(holder["result"])
        assert result["status"] == "answered"
        assert result["answer"] == "English"
        assert result["waited_seconds"] >= 0
    finally:
        _unregister_subagent(sid)


def test_the_parent_is_told_the_child_is_parked_and_how_to_answer(clean_queue, parent_session):
    sid, parent, _child = _child_pair("sa-0-ask0002")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="Which of the two sources is authoritative?",
                              options=["vendor docs", "the repo"], wait_seconds=10)
        entry = _await_pending()

        event = clean_queue.get_nowait()
        assert event["type"] == "subagent_question"
        assert event["question_id"] == entry.question_id
        assert event["session_key"] == "desktop:conv-1"
        assert event["parent_session_id"] == "conv-1"

        text = format_process_notification(event)
        assert "Subagent question from" in text
        assert "PARKED" in text
        assert "Which of the two sources is authoritative?" in text
        assert "vendor docs / the repo" in text
        assert f"question_id='{entry.question_id}'" in text
        assert "gives up at" in text

        questions.answer_pending_question(sid, entry.question_id, "the repo", parent)
        thread.join(timeout=5)
    finally:
        _unregister_subagent(sid)


def test_the_deadline_returns_instructions_instead_of_an_answer(clean_queue, parent_session):
    sid, _parent, _child = _child_pair("sa-0-ask0003")
    try:
        result = json.loads(questions.ask_parent_tool(question="Pick one", wait_seconds=1, task_id=sid))
        assert result["status"] == "timeout"
        assert "stop and return blocked" in result["note"]
        with questions._pending_lock:
            assert not questions._pending, "the rendezvous must be cleaned up on timeout"
    finally:
        _unregister_subagent(sid)


def test_a_stop_request_ends_the_wait(clean_queue, parent_session):
    """The parent can stop a parked child: the wait notices and returns instead of sitting out its budget."""
    sid, _parent, child = _child_pair("sa-0-ask0004")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="Waiting on you", wait_seconds=30)
        _await_pending()
        child._delegate_stop_requested = True
        thread.join(timeout=10)
        assert json.loads(holder["result"])["status"] == "cancelled"
    finally:
        _unregister_subagent(sid)


def test_a_second_question_while_one_is_parked_is_refused(clean_queue, parent_session):
    sid, _parent, _child = _child_pair("sa-0-ask0005")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="First", wait_seconds=10)
        _await_pending()
        second = json.loads(questions.ask_parent_tool(question="Second", task_id=sid))
        assert "already waiting for an answer" in second["error"]
        with questions._pending_lock:
            questions._pending.clear()  # release the parked thread for teardown
        thread.join(timeout=10)
    finally:
        _unregister_subagent(sid)


def test_a_non_child_cannot_ask(clean_queue, parent_session):
    result = json.loads(questions.ask_parent_tool(question="anything", task_id="sa-0-nobody"))
    assert "only available to a running subagent" in result["error"]
    assert clean_queue.empty()


def test_an_unanswerable_question_is_refused_precisely(clean_queue, parent_session):
    sid, parent, _child = _child_pair("sa-0-ask0006")
    try:
        unknown = json.loads(questions.answer_pending_question(sid, "no-such-question", "x", parent))
        assert "not waiting for an answer" in unknown["error"]

        holder: dict = {}
        thread = _parked_call(sid, holder, question="Waiting", wait_seconds=10)
        entry = _await_pending()
        foreign = json.loads(questions.answer_pending_question(sid, entry.question_id, "x", _Parent("other-conv")))
        assert "belongs to another conversation" in foreign["error"]

        questions.answer_pending_question(sid, entry.question_id, "ok", parent)
        thread.join(timeout=5)
    finally:
        _unregister_subagent(sid)


def test_one_pending_question_needs_no_question_id(clean_queue, parent_session):
    """The common case is one question; the id is only required when it is ambiguous."""
    sid, parent, _child = _child_pair("sa-0-ask0007")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="Language?", wait_seconds=10)
        _await_pending()
        receipt = json.loads(questions.answer_pending_question(sid, None, "English", parent))
        assert receipt["status"] == "delivered"
        thread.join(timeout=5)
        assert json.loads(holder["result"])["answer"] == "English"
    finally:
        _unregister_subagent(sid)


def test_list_surfaces_the_pending_question(clean_queue, parent_session):
    """A parent that scrolled past the question can still find it."""
    from tools.delegate_tool_registry import _handle_control_action

    sid, parent, _child = _child_pair("sa-0-ask0008")
    holder: dict = {}
    try:
        thread = _parked_call(sid, holder, question="Blocked on you", wait_seconds=10)
        entry = _await_pending()
        payload = json.loads(_handle_control_action("list", None, None, parent))
        row = next(e for e in payload["subagents"] if e["subagent_id"] == sid)
        assert row["pending_question"] == entry.question_id

        questions.answer_pending_question(sid, entry.question_id, "done", parent)
        thread.join(timeout=5)
    finally:
        _unregister_subagent(sid)


def test_the_wait_cannot_outlast_the_childs_inactivity_budget(monkeypatch):
    """Parking past the delegation watchdog turns a clean timeout into a kill, so the wait is capped."""
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: 120)
    assert questions._wait_budget(3600) == 60
    assert questions._wait_budget(30) == 30
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    assert questions._wait_budget(None) == questions.DEFAULT_WAIT_SECONDS


def test_the_activity_clock_is_pinged_while_parked(clean_queue, parent_session, monkeypatch):
    """The parked call has to look alive to the inactivity tracker, or a slow answer kills the child."""
    pings = {"n": 0}

    def _fake_heartbeat(_label):
        def _tick():
            pings["n"] += 1
        return _tick

    monkeypatch.setattr("tools.approval_human_wait.activity_heartbeat", _fake_heartbeat)
    sid, _parent, _child = _child_pair("sa-0-ask0009")
    try:
        result = json.loads(questions.ask_parent_tool(question="Nobody will answer", wait_seconds=2, task_id=sid))
        assert result["status"] == "timeout"
        assert pings["n"] >= 1
    finally:
        _unregister_subagent(sid)
