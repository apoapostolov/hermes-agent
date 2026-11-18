"""``report_to_parent``: a subagent's mid-run message to its parent agent.

The channel is one event on the shared ``completion_queue``, stamped with the PARENT's session
identity so every poller (CLI, gateway, desktop) treats it as that conversation's notification.
These tests cover the shape the parent model is handed, the routing keys ownership matches on,
and the three ways the tool must fail closed: a caller that is not a child, the per-run cap, and
a host that cannot deliver a later message at all.
"""

import json
import time
import weakref

import pytest

from tools import subagent_message_tool as comms
from tools.delegate_tool import _register_subagent, _unregister_subagent
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification


class _Parent:
    def __init__(self):
        self.session_id = "sess-parent"
        self._current_task_id = "parent-turn-1"


class _Child:
    def __init__(self, parent):
        self._delegate_parent_ref = weakref.ref(parent)


def _child_pair(sid, goal="sweep the audit list"):
    """Register a child and return ``(sid, parent)``.

    The returned parent has to stay referenced for the test's lifetime: a child holds only a
    weakref to it, exactly as a real spawn does, so an unreferenced ``_Parent()`` is collected
    before the call and the tool correctly reports that the caller has no live parent.
    """
    parent = _Parent()
    _register_subagent({"subagent_id": sid, "parent_id": None, "depth": 0, "goal": goal,
                        "model": "m", "started_at": time.time(), "status": "running",
                        "tool_count": 0, "agent": _Child(parent),
                        "owner_agent_session_id": parent.session_id})
    return sid, parent



@pytest.fixture(autouse=True)
def _plain_spawn(monkeypatch):
    """Spawn plain children: the systemd-run --user --scope wrapper is irrelevant here and stalls under pytest."""
    import tools.process_registry as _pr
    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)


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
    """Pin the identity a child inherits from its parent's conversation, and a deliverable host."""
    from gateway import session_context

    env = {"HERMES_SESSION_KEY": "desktop:conv-1", "HERMES_SESSION_ID": "conv-1",
           "HERMES_SESSION_PLATFORM": "desktop"}
    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": env.get(key, default))
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)

    def _no_delivery():
        return False
    return monkeypatch.setattr


def test_child_message_is_queued_with_the_parents_routing(clean_queue, parent_session):
    """The event carries the parent's conversation identity, the child's id and goal, and the text."""
    sid, _parent = _child_pair("sa-0-msg0001")
    try:
        receipt = json.loads(comms.report_to_parent_tool("Half the audit list is stale.", kind="finding", task_id=sid))
        assert receipt.get("status") == "queued", receipt
        assert receipt["messages_remaining"] == comms.MAX_MESSAGES_PER_CHILD - 1

        event = clean_queue.get_nowait()
        assert event["type"] == "subagent_message"
        assert event["subagent_id"] == sid
        assert event["goal"] == "sweep the audit list"
        assert event["kind"] == "finding"
        # The parent's identity, not the child's: ownership and the closed-chat pre-flight read these.
        assert event["session_key"] == "desktop:conv-1"
        assert event["parent_session_id"] == "conv-1"
        assert event["owner_task_id"] == "sess-parent"
        assert event["text"] == "Half the audit list is stale."
    finally:
        _unregister_subagent(sid)


def test_formatter_tells_the_parent_who_sent_it_and_how_to_answer(clean_queue, parent_session):
    """The parent sees the sender, its goal, the message, and the steer/stop verbs it can use."""
    sid, _parent = _child_pair("sa-0-msg0002")
    try:
        comms.report_to_parent_tool("Found the cause: a stale cache key.", task_id=sid)
        text = format_process_notification(clean_queue.get_nowait())

        assert f"Subagent message from {sid}" in text
        assert "sweep the audit list" in text
        assert "Found the cause: a stale cache key." in text
        assert "still working" in text
        assert "action='steer'" in text
        assert "action='stop'" in text
    finally:
        _unregister_subagent(sid)


def test_a_non_child_cannot_use_the_channel(clean_queue, parent_session):
    """A top-level agent, or a fabricated id in prose, is refused rather than silently delivered."""
    result = json.loads(comms.report_to_parent_tool("hello", task_id="sa-0-notmine"))
    assert "error" in result
    assert "only available to a running subagent" in result["error"]
    assert clean_queue.empty()


def test_the_per_run_cap_refuses_and_says_where_to_put_the_rest(clean_queue, parent_session):
    """Every delivered message costs the parent a turn, so the cap is a guardrail and not a detail."""
    sid, _parent = _child_pair("sa-0-msg0003")
    try:
        for _ in range(comms.MAX_MESSAGES_PER_CHILD):
            assert json.loads(comms.report_to_parent_tool("step done", task_id=sid))["status"] == "queued"
        result = json.loads(comms.report_to_parent_tool("one more", task_id=sid))
        assert "cap reached" in result["error"]
        assert "final summary" in result["error"]
    finally:
        _unregister_subagent(sid)


def test_a_host_that_cannot_deliver_later_refuses(clean_queue, monkeypatch):
    """One-shot and worker sessions have no reader for the queue: refuse instead of queueing a dead message."""
    from gateway import session_context

    monkeypatch.setattr(session_context, "get_session_env", lambda key, default="": "")
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)
    sid, _parent = _child_pair("sa-0-msg0004")
    try:
        result = json.loads(comms.report_to_parent_tool("anything", task_id=sid))
        assert "cannot deliver a mid-run message" in result["error"]
        assert clean_queue.empty()
    finally:
        _unregister_subagent(sid)


def test_an_empty_message_is_refused(clean_queue, parent_session):
    sid, _parent = _child_pair("sa-0-msg0005")
    try:
        assert "error" in json.loads(comms.report_to_parent_tool("   ", task_id=sid))
        assert clean_queue.empty()
    finally:
        _unregister_subagent(sid)


def test_delegated_children_are_granted_the_toolset():
    """The toolset is role-granted to children, so no interactive session has to enable it."""
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    enabled, disabled = _resolve_child_toolsets(None, None, "leaf")
    assert "subagent_comms" in enabled
    assert "subagent_comms" not in disabled
