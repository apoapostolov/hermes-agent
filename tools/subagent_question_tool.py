#!/usr/bin/env python3
"""``ask_parent``: a subagent that blocks on a decision instead of guessing.

``report_to_parent`` is one-way: the child says something and keeps working. A question needs
the answer back *inside the tool call*, and three properties of this engine make that harder
than it looks:

* ``delegate_task(action='steer')`` is delivered after the child's next **tool result**, which is
  the exact boundary a parked child no longer crosses, so a steer-only answer would arrive as a
  duplicate or never.
* The parent is usually idle while a child runs, and delivery waits for an idle window (measured:
  ~13 minutes late on a busy thread). The round trip is slow by construction, so a timeout is
  part of the contract rather than an edge case.
* ``delegation.child_timeout_seconds`` is an **inactivity** cap. A child parked in a tool call for
  the whole wait would read as frozen and be warned, then killed, unless the wait keeps the
  activity clock ticking.

So the design is a rendezvous: the tool registers a pending question, puts a
``type="subagent_question"`` event on the shared completion queue, then parks on a
``threading.Event`` while ``tools.approval_human_wait.activity_heartbeat`` keeps the inactivity
tracker warm. ``delegate_task(action='answer', subagent_id=..., question_id=..., message=...)``
resolves the rendezvous and the parked tool returns the answer as its result.

Ownership is the same spine every other control verb uses (``_owns_subagent_record``), so only the
conversation that spawned the child can answer it, and only for a question that is still pending.
The wait never exceeds the child's own inactivity budget minus a margin, because the alternative
is a race the watchdog wins by killing the child mid-question.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Default wait when the child does not ask for one. Long enough for a user who stepped away to
# come back, short enough that a child is not parked for a whole afternoon.
DEFAULT_WAIT_SECONDS = 900
# Never wait closer than this to the child's inactivity deadline: the watchdog polls in slices and
# a wait that ends level with it turns a clean timeout into a kill.
_WATCHDOG_MARGIN_SECONDS = 60
# How often the parked call re-checks the answer, the stop flag, and the activity clock.
_WAIT_SLICE_SECONDS = 1.0
_MAX_QUESTIONS_PER_CHILD = 3
_MAX_QUESTION_CHARS = 2_000
_MAX_OPTIONS = 4
_MAX_OPTION_CHARS = 120

_TIMEOUT_NOTE = (
    "No answer arrived before the deadline and nothing is listening for it now. Do not invent one "
    "for anything that spends money, publishes, deletes or messages someone: stop and return "
    "blocked, saying what you need. For everything else, take the reading your brief implies, "
    "carry on, and name the assumption you made in your final summary."
)
_CANCELLED_NOTE = "The parent stopped you while you waited. Stop now and return what you have."


@dataclass
class PendingQuestion:
    """One parked child: the question it asked, and the slot its answer lands in."""

    question_id: str
    subagent_id: str
    text: str
    options: list[str]
    event: threading.Event = field(default_factory=threading.Event)
    answer: Optional[str] = None
    asked_at: float = field(default_factory=time.time)
    deadline: float = 0.0

    def resolve(self, answer: str) -> None:
        self.answer = answer
        self.event.set()


_pending: dict[str, PendingQuestion] = {}
_pending_lock = threading.Lock()


def _prune_expired(now: Optional[float] = None) -> None:
    """Drop questions whose deadline passed unanswered.

    A child that was killed while parked never runs its own cleanup, so every entry point prunes
    rather than relying on a teardown callback. Pruning a live question cannot happen: a parked
    child's deadline IS its wake-up time.
    """
    moment = time.time() if now is None else now
    with _pending_lock:
        stale = [qid for qid, q in _pending.items() if q.deadline and q.deadline <= moment]
        for qid in stale:
            _pending.pop(qid, None)
    for qid in stale:
        logger.debug("pruned expired subagent question %s", qid)


def pending_questions(subagent_id: Optional[str] = None) -> list[PendingQuestion]:
    """Live questions, optionally for one child; used by ``delegate_task(action='list')``."""
    _prune_expired()
    with _pending_lock:
        questions = list(_pending.values())
    if subagent_id:
        questions = [q for q in questions if q.subagent_id == subagent_id]
    return questions


def _wait_budget(requested: Optional[int]) -> int:
    """The longest a child may park: its request, capped below its own inactivity deadline."""
    from tools.delegate_tool_config import _get_child_timeout

    try:
        wait = int(requested) if requested else DEFAULT_WAIT_SECONDS
    except (TypeError, ValueError):
        wait = DEFAULT_WAIT_SECONDS
    wait = max(1, wait)
    child_timeout = _get_child_timeout()
    if child_timeout:
        ceiling = int(child_timeout) - _WATCHDOG_MARGIN_SECONDS
        if ceiling > 0:
            wait = min(wait, ceiling)
    return wait


def _question_event(record: dict, text: str, options: list[str], question_id: str, deadline: float) -> dict:
    """The queue event: same routing stamp as a message, plus what an answerer needs."""
    from gateway.session_context import get_session_env

    parent_owner = str(record.get("owner_agent_session_id") or "")
    return {
        "type": "subagent_question",
        "event_id": uuid.uuid4().hex,
        "session_id": "",
        "session_key": get_session_env("HERMES_SESSION_KEY", ""),
        "parent_session_id": get_session_env("HERMES_SESSION_ID", "") or parent_owner,
        "task_id": parent_owner,
        "owner_task_id": parent_owner,
        "subagent_id": str(record.get("subagent_id") or ""),
        "goal": " ".join(str(record.get("goal") or "").split())[:120],
        "question_id": question_id,
        "text": text,
        "options": options,
        "timeout_at": deadline,
        "created_at": time.time(),
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
    }


def ask_parent_tool(
    question: str, options: Optional[list[str]] = None, wait_seconds: Optional[int] = None,
    task_id: Optional[str] = None,
) -> str:
    """Ask the parent a question and park until it answers, the deadline passes, or the child is stopped."""
    text = " ".join(str(question or "").split())
    if not text:
        return tool_error("ask_parent needs `question`: state the decision you cannot take, in one or two sentences.")
    if len(text) > _MAX_QUESTION_CHARS:
        text = text[:_MAX_QUESTION_CHARS].rstrip() + " …(truncated)"

    from tools.subagent_message_tool import _caller_child

    record, child, error = _caller_child(task_id)
    if error:
        return tool_error(error.replace("report_to_parent", "ask_parent"))

    asked = getattr(child, "_parent_questions", None)
    if asked is None:
        asked = child._parent_questions = []
    if len(asked) >= _MAX_QUESTIONS_PER_CHILD:
        return tool_error(
            f"ask_parent cap reached ({_MAX_QUESTIONS_PER_CHILD} questions this run); two rounds of "
            "questions is already a lot to spend a run on. Decide the rest or return blocked."
        )
    if pending_questions(record.get("subagent_id")):
        # A child can only be parked in one tool call, so a second concurrent question is a mistake.
        return tool_error("a question from you is already waiting for an answer; one at a time.")

    from gateway.session_context import async_delivery_supported

    if not async_delivery_supported():
        return tool_error(
            "this host cannot deliver a question later (one-shot or worker session); decide within your "
            "brief, or return blocked in your final summary."
        )

    cleaned_options = [str(o).strip()[:_MAX_OPTION_CHARS] for o in (options or []) if str(o).strip()][:_MAX_OPTIONS]
    budget = _wait_budget(wait_seconds)
    question_id = uuid.uuid4().hex
    entry = PendingQuestion(
        question_id=question_id,
        subagent_id=str(record.get("subagent_id") or ""),
        text=text,
        options=cleaned_options,
        deadline=time.time() + budget,
    )
    event = _question_event(record, text, cleaned_options, question_id, entry.deadline)
    try:
        from tools.delegation_redaction import redact_completion_event
        event = redact_completion_event(event)
    except Exception:
        logger.debug("subagent question redaction failed", exc_info=True)
        return tool_error("ask_parent could not redact this question; put it in your final summary instead.")

    with _pending_lock:
        _pending[question_id] = entry
    asked.append({"question_id": question_id, "at": entry.asked_at})

    from tools.process_registry import process_registry

    process_registry.completion_queue.put(event)

    from tools.approval_human_wait import activity_heartbeat

    heartbeat = activity_heartbeat("subagent waiting for its parent's answer")
    started = time.monotonic()
    try:
        while True:
            # Waits in slices: the answer can arrive at any moment, and a stop request has to be
            # noticed while parked or the child would sit out its own cancellation.
            if entry.event.wait(_WAIT_SLICE_SECONDS):
                break
            heartbeat()
            if getattr(child, "_delegate_stop_requested", False):
                return json.dumps({
                    "status": "cancelled",
                    "note": _CANCELLED_NOTE,
                }, ensure_ascii=False)
            if time.monotonic() - started >= budget:
                return json.dumps({
                    "status": "timeout",
                    "waited_seconds": round(time.monotonic() - started, 1),
                    "note": _TIMEOUT_NOTE,
                }, ensure_ascii=False)
    finally:
        with _pending_lock:
            _pending.pop(question_id, None)

    waited = round(time.monotonic() - started, 1)
    logger.debug("subagent %s received an answer after %ss", entry.subagent_id, waited)
    return json.dumps({
        "status": "answered",
        "answer": entry.answer or "",
        "waited_seconds": waited,
        "note": "The parent's answer. Act on it, and say in your summary which decision it settled.",
    }, ensure_ascii=False)


def answer_pending_question(
    subagent_id: Optional[str], question_id: Optional[str], answer: Optional[str], parent_agent: Any,
) -> str:
    """Resolve one pending question. Ownership is the same spine as steer/stop."""
    from tools.delegate_tool_registry import _owns_subagent_record, _active_subagents, _active_subagents_lock

    sid = (subagent_id or "").strip()
    if not sid:
        return tool_error("action='answer' requires subagent_id (from action='list' or the question message).")
    body = " ".join(str(answer or "").split())
    if not body:
        return tool_error("action='answer' requires `message`: the decision, stated as the child should act on it.")

    qid = (question_id or "").strip()
    if not qid:
        # One pending question is the common case; only an ambiguous one needs the id spelled out.
        candidates = pending_questions(sid)
        if len(candidates) == 1:
            qid = candidates[0].question_id
        elif not candidates:
            return tool_error(f"no question from '{sid}' is waiting for an answer (already answered, timed out, or never asked).")
        else:
            return tool_error(
                f"'{sid}' has {len(candidates)} questions waiting; pass question_id to choose one "
                f"({', '.join(q.question_id for q in candidates)})."
            )

    with _pending_lock:
        entry = _pending.get(qid)
    if entry is None:
        return tool_error(f"question '{qid}' is not waiting for an answer (already answered, timed out, or withdrawn).")
    if entry.subagent_id != sid:
        return tool_error(f"question '{qid}' belongs to '{entry.subagent_id}', not '{sid}'.")

    with _active_subagents_lock:
        record = _active_subagents.get(sid)
    if record is None:
        return tool_error(f"'{sid}' is no longer a live subagent; its question cannot be answered.")
    if not _owns_subagent_record(record, parent_agent):
        return tool_error(f"'{sid}' belongs to another conversation; this one cannot answer it.")

    entry.resolve(body)
    return json.dumps({
        "action": "answer",
        "subagent_id": sid,
        "question_id": qid,
        "status": "delivered",
        "note": "The child is unparked and resumed with this answer as its tool result.",
    }, ensure_ascii=False)


ASK_PARENT_SCHEMA = {
    "name": "ask_parent",
    "description": (
        "Ask your parent agent a question and WAIT for the answer inside this call. Use it when the "
        "next step depends on a decision only the user can make: a trade-off, a permission, a "
        "content choice. The parent may be idle, so the wait can be minutes; you are parked, not "
        "stopped, and your place in the work is kept. Provide `options` when there are a few clean "
        "choices so the parent can put a one-line question to the user. If the deadline passes you "
        "get a timeout result telling you what to do next. Do not use this for anything you can "
        "settle from your brief or the repository: every question costs the user a decision."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The decision, stated so it can be answered without reading your transcript. "
                    "Include the two or three facts that make it a real choice."
                ),
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_OPTIONS,
                "description": "Optional short answers (e.g. 'Bulgarian', 'English'), at most four.",
            },
            "wait_seconds": {
                "type": "integer",
                "description": (
                    f"How long to wait before giving up; default {DEFAULT_WAIT_SECONDS}. The effective "
                    "wait is capped below your own inactivity budget."
                ),
            },
        },
        "required": ["question"],
    },
}


registry.register(
    name="ask_parent",
    toolset="subagent_comms",
    schema=ASK_PARENT_SCHEMA,
    handler=lambda args, **kw: ask_parent_tool(
        question=args.get("question", ""),
        options=args.get("options"),
        wait_seconds=args.get("wait_seconds"),
        task_id=kw.get("task_id"),
    ),
    emoji="❓",
)
