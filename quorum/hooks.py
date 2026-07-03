"""Extension hooks around a deliberation.

Lets new stages attach to :func:`orchestrator.run_session` without editing it --
the extensibility lever for features that are *stages* rather than *knobs*
(retrieval, safety/PII filters, answer post-processing, guardrails):

* **pre-hooks** run after the :class:`~quorum.strategies.Context` is built and
  *before* the strategy. They may read or mutate ``ctx`` (e.g. enrich
  ``ctx.prompt`` with retrieved context, or adjust ``ctx.opts``).
* **post-hooks** run *after* the strategy and before persistence. They may read
  or adjust the finished session via ``ctx.session`` (e.g. redact/annotate
  ``ctx.session.final``).

Both are empty by default, so a stock engine behaves exactly as before. In-tree
code registers with :func:`register_pre` / :func:`register_post`; out-of-tree
packages can register via the ``quorum.hooks.pre`` / ``quorum.hooks.post``
entry-point groups (discovered once, lazily).
"""
from __future__ import annotations

from typing import Any, Callable

Hook = Callable[[Any], None]   # (ctx) -> None

_PRE: list[Hook] = []
_POST: list[Hook] = []
_ep_loaded = False


def register_pre(fn: Hook) -> Hook:
    """Register a pre-deliberation hook. Returns ``fn`` so it works as a decorator."""
    _PRE.append(fn)
    return fn


def register_post(fn: Hook) -> Hook:
    """Register a post-deliberation hook. Returns ``fn`` so it works as a decorator."""
    _POST.append(fn)
    return fn


def clear() -> None:
    """Drop all registered hooks (and re-arm entry-point discovery) -- for tests."""
    global _ep_loaded
    _PRE.clear()
    _POST.clear()
    _ep_loaded = False


def _load_entry_points() -> None:
    global _ep_loaded
    if _ep_loaded:
        return
    _ep_loaded = True
    for group, sink in (("quorum.hooks.pre", _PRE), ("quorum.hooks.post", _POST)):
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            selected = eps.select(group=group) if hasattr(eps, "select") \
                else eps.get(group, [])  # py<3.10 shape
            sink.extend(ep.load() for ep in selected)
        except Exception:  # noqa: BLE001 - hooks are optional, never fatal
            pass


def run_pre(ctx: Any) -> None:
    _load_entry_points()
    for fn in _PRE:
        fn(ctx)


def run_post(ctx: Any) -> None:
    _load_entry_points()
    for fn in _POST:
        fn(ctx)
