"""Plain-text rendering of a deliberation session (for the CLI ``run``/``show``)."""
from __future__ import annotations

from typing import Any

from .model import Session

_KIND_GLYPH = {
    "promptsmith": "*", "propose": ".", "revise": "~", "review": "?",
    "synthesize": "=", "aggregate": "=", "judge": "#",
}


def render_session(session: Session) -> str:
    return render_session_dict(session.to_dict())


def render_session_dict(d: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"session {d['id']}   strategy={d['strategy']}   "
                 f"score={d['final_score']:.1f}   cost=${d['cost_usd']:.4f}   "
                 f"tokens={d['tokens_in'] + d['tokens_out']}")
    lines.append(f"task: {d['task']}")
    if d.get("prompt") and d["prompt"].strip() != d["task"].strip():
        lines.append(f"refined prompt: {_clip(d['prompt'], 200)}")
    lines.append("")

    for rnd in d.get("rounds", []):
        idx = rnd["index"]
        head = "promptsmith" if idx == 0 else f"round {idx}"
        v = rnd.get("verdict")
        if v:
            head += f"   score={v['score']:.1f}  best={v['best_label']}"
        lines.append(f"-- {head} " + "-" * max(0, 40 - len(head)))
        for t in rnd.get("turns", []):
            g = _KIND_GLYPH.get(t["kind"], "-")
            lines.append(f"  {g} [{t['member']}/{t['kind']}] {_clip(t['content'], 150)}")
        if v and v.get("rationale"):
            lines.append(f"    judge: {_clip(v['rationale'], 150)}")
        lines.append("")

    lines.append(f"stop: {d.get('stop_reason', '')}   status: {d.get('status', 'ok')}")
    lines.append("\n=== FINAL ANSWER ===\n" + (d.get("final") or "(none)"))
    return "\n".join(lines)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "\u2026"
