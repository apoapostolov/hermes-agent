#!/usr/bin/env python3
"""``report_to_parent``: a running subagent's mid-run route to its parent agent.

A child is stripped of every tool that could address the user or its parent (``clarify``,
``send_message``, ``delegate_task``), and the background processes it starts may not notify
either (``tools/terminal_tool_background.py`` forces ``notify_on_complete`` off for a
subagent). That leaves a child with something worth saying mid-run no route at all: its only
escalation is to finish, which loses whatever it was in the middle of.

This tool writes one ``type="subagent_message"`` event onto the shared
``process_registry.completion_queue``. The CLI, the gateway and the desktop poller each drain
that queue while idle and turn an event into a new message in the parent's conversation
(``tui_gateway/session_notifications.py::_notif_handle_event``), so the parent learns what the
child found without the child stopping, and can answer with
``delegate_task(action='steer', subagent_id=..., message=...)`` or leave it alone.

Guardrails, all deliberate:

* **Child-only.** The caller must be a live delegated child in this conversation's spawn tree,
  checked the same way ``process_manage(action='handoff')`` checks it, so prose about a subagent
  id cannot masquerade as one.
* **No addressing.** The event borrows the *parent's* routing identity from the session context.
  A child cannot name a session, chat, thread or user, so it cannot message anyone else.
* **Bounded.** ``MAX_MESSAGES_PER_CHILD`` per run and ``MAX_MESSAGE_CHARS`` per message: every
  delivered event costs the parent a whole model turn, which is the expensive half of this
  feature.
* **Fail closed.** A host that cannot deliver later (one-shot, Kanban worker, managed worker
  process) gets a refusal telling the child to put the finding in its final summary, rather than
  a message that is queued and never read.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# One delivered message costs the parent a full turn, so the cap is the point rather than a
# detail: a child that narrates every step would spend the parent's context on its own.
MAX_MESSAGES_PER_CHILD = 8
MAX_MESSAGE_CHARS = 4_000
_MAX_GOAL_CHARS = 120
_KINDS = ("progress", "question", "blocked", "finding")

_NOT_A_CHILD = (
    "report_to_parent is only available to a running subagent with a live parent; you are not "
    "one. Answer the user in your final summary instead."
)


def _clip(text: Any, limit: int) -> str:
    """Collapse to one line and truncate, saying how much was dropped."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + " …"


def _caller_child(task_id: Optional[str]) -> tuple[Optional[dict], Any, str]:
    """``(record, child, error)`` for the calling subagent: the spawn-tree check handoff uses."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    with _active_subagents_lock:
        record = _active_subagents.get(str(task_id or ""))
    child = record.get("agent") if record else None
    if child is None:
        return None, None, _NOT_A_CHILD
    parent_ref = getattr(child, "_delegate_parent_ref", None)
    parent = parent_ref() if callable(parent_ref) else None
    if parent is None:
        # A child outlives its parent only while being torn down; there is no one to tell.
        return None, None, _NOT_A_CHILD
    return record, child, ""


def _build_event(record: dict, text: str, kind: str, seq: int) -> dict:
    """One queue event, routed exactly like a process notification from the same conversation.

    The routing keys are the parent's, read from the session context the child inherits:
    ``session_key`` is what ``_session_owns_notification_event`` matches on, and
    ``parent_session_id`` is what the gateway's completion pre-flight uses to drop a message
    whose conversation the user closed (``/new``) in the meantime.
    """
    from gateway.session_context import get_session_env

    parent_owner = str(record.get("owner_agent_session_id") or "")
    return {
        "type": "subagent_message",
        "event_id": uuid.uuid4().hex,
        "session_id": "",
        "session_key": get_session_env("HERMES_SESSION_KEY", ""),
        "parent_session_id": get_session_env("HERMES_SESSION_ID", "") or parent_owner,
        "task_id": parent_owner,
        "owner_task_id": parent_owner,
        "subagent_id": str(record.get("subagent_id") or ""),
        "goal": _clip(record.get("goal"), _MAX_GOAL_CHARS),
        "kind": kind,
        "seq": seq,
        "text": text,
        "created_at": time.time(),
        "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
    }


def report_to_parent_tool(message: str, kind: str = "progress", task_id: Optional[str] = None) -> str:
    """Queue one mid-run message for the parent agent; returns the delivery receipt as JSON."""
    text = " ".join(str(message or "").split())
    if not text:
        return tool_error("report_to_parent needs `message`: one or two sentences the parent should know now.")
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS].rstrip() + f" …(+{len(text) - MAX_MESSAGE_CHARS} chars truncated)"

    record, child, error = _caller_child(task_id)
    if error:
        return tool_error(error)

    sent = getattr(child, "_parent_messages", None)
    if sent is None:
        sent = child._parent_messages = []
    if len(sent) >= MAX_MESSAGES_PER_CHILD:
        return tool_error(
            f"report_to_parent cap reached ({MAX_MESSAGES_PER_CHILD} messages this run); the parent has "
            "everything you sent. Put anything further in your final summary."
        )

    from gateway.session_context import async_delivery_supported

    if not async_delivery_supported():
        return tool_error(
            "this host cannot deliver a mid-run message later (one-shot or worker session); put the "
            "finding in your final summary instead."
        )

    kind = str(kind or "progress").strip().lower()
    if kind not in _KINDS:
        kind = "progress"

    event = _build_event(record, text, kind, len(sent) + 1)
    try:
        from tools.delegation_redaction import redact_completion_event
        event = redact_completion_event(event)
    except Exception:  # a broken redactor must not leak the text it was meant to scrub
        logger.debug("subagent message redaction failed", exc_info=True)
        return tool_error("report_to_parent could not redact this message; put it in your final summary.")

    from tools.process_registry import process_registry

    process_registry.completion_queue.put(event)
    sent.append({"seq": event["seq"], "kind": kind, "at": event["created_at"]})
    logger.debug(
        "subagent %s queued parent message #%s (%s chars, %s)",
        event.get("subagent_id"), event.get("seq"), len(text), kind,
    )
    return json.dumps({
        "status": "queued",
        "seq": event["seq"],
        "messages_sent": len(sent),
        "messages_remaining": MAX_MESSAGES_PER_CHILD - len(sent),
        "note": (
            "Delivered to your parent's conversation at its next idle moment; you keep running. "
            "Keep working unless the parent steers you."
        ),
    }, ensure_ascii=False)


REPORT_TO_PARENT_SCHEMA = {
    "name": "report_to_parent",
    "description": (
        "Send your parent agent a short message from the middle of a delegated run, without "
        "stopping. The parent reads it between turns and can reply with a course correction; you "
        "carry on working either way. Use it for a finding that changes what the parent should do "
        "next, a decision you cannot take on your own, or a blocker that will not clear. It is not "
        "a progress log: the parent already sees your transcript, so each message should say "
        "something it could not see there. Budget: 8 messages per run, 4000 characters each."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "One or two sentences stating what the parent should know or decide. Name the "
                    "concrete thing (path, id, number), not the activity."
                ),
            },
            "kind": {
                "type": "string",
                "enum": list(_KINDS),
                "description": (
                    "Why you are messaging. 'finding' = something the parent's plan should account "
                    "for; 'question' = you need a decision; 'blocked' = you cannot continue without "
                    "help; 'progress' = a milestone the parent needs for its own next step."
                ),
            },
        },
        "required": ["message"],
    },
}


registry.register(
    name="report_to_parent",
    toolset="subagent_comms",
    schema=REPORT_TO_PARENT_SCHEMA,
    handler=lambda args, **kw: report_to_parent_tool(
        message=args.get("message", ""),
        kind=args.get("kind") or "progress",
        task_id=kw.get("task_id"),
    ),
    emoji="📣",
)
