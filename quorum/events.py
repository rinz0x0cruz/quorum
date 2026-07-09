"""Structured deliberation events -- a minimal observability channel.

The orchestrator and strategies emit human log lines *and* optional structured
:class:`Event` objects through the same ``ctx.emit`` sink. A caller can pass
``on_event`` to :func:`quorum.orchestrator.run_session` to subscribe to the typed
stream (progress UIs, richer bench telemetry, live serving) while the CLI keeps
printing the same one-line messages. Plain-string emits are wrapped as ``log``
events, so nothing changes for existing call sites.

Layering: a leaf module -- imports only the stdlib, so anything may depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Event:
    """A structured progress event. ``kind`` is a stable tag; ``message`` is the
    human one-liner the CLI prints; ``data`` carries the structured payload."""

    kind: str                                   # log | phase | round | member_failed | done
    message: str = ""
    round: int = 0
    data: dict = field(default_factory=dict)


def render(event: Event) -> str:
    """The human line for an event (what the verbose CLI prints)."""
    return event.message or event.kind


def coerce(x: Union[Event, str]) -> Event:
    """Wrap a plain-string emit as a ``log`` event (backward compatibility)."""
    return x if isinstance(x, Event) else Event("log", str(x))
