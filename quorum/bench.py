"""Benchmark harness: run several strategies over a task set and compare them.

For each (strategy, task) it runs a full deliberation and records the judge score,
rounds, tokens, cost, and wall-time. It then aggregates per strategy -- mean
score, win-rate (how often a strategy is the best on a task), and mean
cost/tokens/rounds/time -- so you can see empirically which strategy is worth its
cost on *your* tasks. Runs offline against the ``mock`` provider.

Task file (YAML or JSON), either a bare list or under a ``tasks:`` key::

    tasks:
      - id: capital
        task: "What is the capital of Australia? Explain the common mistake."
      - "A bare string is also a valid task."
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

from . import orchestrator


def run(cfg: dict, tasks_path: str, strategies: list[str], store: Any, *,
        as_json: bool = False, verbose: bool = True) -> int:
    tasks = _load_tasks(tasks_path)
    if not tasks:
        print(f"  no tasks found in {tasks_path}")
        return 1

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for strat in strategies:
            rcfg = copy.deepcopy(cfg)
            if task.get("rubric"):
                rcfg.setdefault("judge", {})["rubric"] = task["rubric"]
            t0 = time.time()
            sess = orchestrator.run_session(rcfg, task["task"], store=store,
                                            strategy=strat, verbose=False)
            secs = time.time() - t0
            row = {
                "strategy": strat, "task_id": task["id"], "score": round(sess.final_score, 2),
                "rounds": len([r for r in sess.rounds if r.index != 0]),
                "tokens_in": sess.tokens_in, "tokens_out": sess.tokens_out,
                "tokens": sess.tokens_in + sess.tokens_out,
                "cost_usd": round(sess.cost_usd, 6), "seconds": round(secs, 3),
            }
            rows.append(row)
            store.add_bench_row(strat, task["id"], row["score"], row["rounds"],
                                row["tokens_in"], row["tokens_out"], row["cost_usd"], row["seconds"])
            if verbose:
                print(f"  {task['id']:<14} {strat:<9} score={row['score']:>5.1f}  "
                      f"rounds={row['rounds']}  tokens={row['tokens']:>5}  "
                      f"cost=${row['cost_usd']:.4f}  {row['seconds']:.2f}s")

    summary = aggregate(rows, strategies, len(tasks))
    if as_json:
        print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    else:
        print("\n" + _table(summary))
    store.add_run("bench", len(rows), "ok")
    return 0


def aggregate(rows: list[dict[str, Any]], strategies: list[str], n_tasks: int) -> list[dict[str, Any]]:
    # wins: per task, the strategy with the highest score (ties -> shared).
    wins: dict[str, float] = {s: 0.0 for s in strategies}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    for task_rows in by_task.values():
        best = max(r["score"] for r in task_rows)
        leaders = [r for r in task_rows if r["score"] == best]
        for r in leaders:
            wins[r["strategy"]] += 1.0 / len(leaders)

    out = []
    for s in strategies:
        sr = [r for r in rows if r["strategy"] == s]
        if not sr:
            continue
        k = len(sr)
        out.append({
            "strategy": s,
            "mean_score": round(sum(r["score"] for r in sr) / k, 2),
            "win_rate": round(100.0 * wins[s] / max(1, n_tasks), 1),
            "mean_rounds": round(sum(r["rounds"] for r in sr) / k, 2),
            "mean_tokens": int(sum(r["tokens"] for r in sr) / k),
            "mean_cost_usd": round(sum(r["cost_usd"] for r in sr) / k, 6),
            "mean_seconds": round(sum(r["seconds"] for r in sr) / k, 3),
        })
    out.sort(key=lambda r: (r["mean_score"], r["win_rate"]), reverse=True)
    return out


def _table(summary: list[dict[str, Any]]) -> str:
    head = f"{'strategy':<10} {'score':>6} {'win%':>6} {'rounds':>7} {'tokens':>7} {'cost$':>9} {'sec':>6}"
    lines = [head, "-" * len(head)]
    for r in summary:
        lines.append(f"{r['strategy']:<10} {r['mean_score']:>6.1f} {r['win_rate']:>6.1f} "
                     f"{r['mean_rounds']:>7.2f} {r['mean_tokens']:>7} "
                     f"{r['mean_cost_usd']:>9.4f} {r['mean_seconds']:>6.2f}")
    if summary:
        lines.append(f"\n  winner: {summary[0]['strategy']} "
                     f"(mean score {summary[0]['mean_score']:.1f}, win-rate {summary[0]['win_rate']:.0f}%)")
    return "\n".join(lines)


def _load_tasks(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json"):
            data = json.load(fh)
        else:
            import yaml
            data = yaml.safe_load(fh)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    tasks = []
    for i, item in enumerate(data or []):
        if isinstance(item, str):
            tasks.append({"id": f"t{i + 1}", "task": item})
        elif isinstance(item, dict) and item.get("task"):
            tasks.append({"id": str(item.get("id", f"t{i + 1}")), "task": item["task"],
                          "rubric": item.get("rubric")})
    return tasks
