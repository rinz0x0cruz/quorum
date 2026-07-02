"""Export a deliberation session as JSON, CSV, or Markdown.

JSON is the full session blob; CSV is a per-turn flat table for spreadsheets;
Markdown is a readable transcript (task -> refined prompt -> rounds -> final).
Defaults to the most recent session when ``--session`` is omitted.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Optional

from .store import Store


def run(cfg: dict, store: Store, *, fmt: str = "json", session_id: Optional[str] = None,
        out: Optional[str] = None) -> int:
    d = store.get_session(session_id) if session_id else _latest(store)
    if not d:
        print("  no session to export" + (f" ({session_id})" if session_id else ""))
        return 1

    sid = d["id"]
    if fmt == "csv":
        path = out or f"data/{sid}.csv"
        _write_csv(path, d)
    elif fmt == "md":
        path = out or f"data/{sid}.md"
        _write_md(path, d)
    else:
        path = out or f"data/{sid}.json"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
    print(f"  exported -> {path}")
    return 0


def _latest(store: Store) -> Optional[dict[str, Any]]:
    recent = store.recent_sessions(1)
    return recent[0] if recent else None


def _write_csv(path: str, d: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "member", "model", "kind", "tokens_in", "tokens_out",
                    "cost_usd", "score", "content"])
        for r in d.get("rounds", []):
            score = (r.get("verdict") or {}).get("score", "")
            for t in r.get("turns", []):
                w.writerow([r["index"], t["member"], t["model"], t["kind"], t["tokens_in"],
                            t["tokens_out"], t["cost_usd"], score, " ".join((t["content"] or "").split())])


def _write_md(path: str, d: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lines = [f"# quorum session `{d['id']}`", "",
             f"- **strategy**: {d['strategy']}",
             f"- **score**: {d['final_score']:.1f}",
             f"- **cost**: ${d['cost_usd']:.4f} · **tokens**: {d['tokens_in'] + d['tokens_out']}",
             f"- **stop**: {d.get('stop_reason', '')} · **status**: {d.get('status', 'ok')}",
             "", f"## Task", "", d["task"], ""]
    if d.get("prompt") and d["prompt"].strip() != d["task"].strip():
        lines += ["## Refined prompt", "", d["prompt"], ""]
    for r in d.get("rounds", []):
        title = "Promptsmith" if r["index"] == 0 else f"Round {r['index']}"
        v = r.get("verdict")
        if v:
            title += f" — score {v['score']:.1f} (best: {v['best_label']})"
        lines += [f"## {title}", ""]
        for t in r.get("turns", []):
            lines += [f"**{t['member']}** · _{t['kind']}_ · `{t['model']}`", "",
                      (t["content"] or "").strip(), ""]
        if v and v.get("rationale"):
            lines += [f"> judge: {v['rationale']}", ""]
    lines += ["## Final answer", "", d.get("final", "(none)"), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
