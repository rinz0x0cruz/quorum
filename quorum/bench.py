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

from . import grade, orchestrator, provider


def run(cfg: dict, tasks_path: str, strategies: list[str], store: Any, *,
        as_json: bool = False, verbose: bool = True) -> int:
    tasks = _load_tasks(tasks_path)
    if not tasks:
        print(f"  no tasks found in {tasks_path}")
        return 1

    # One shared provider across every task + grading, so a single rate limiter
    # paces the whole bench under the per-minute cap (a fresh provider per task
    # would reset the limiter and let bursts through) and all attempts land in the
    # same throttle telemetry.
    prov = provider.for_config(cfg, store=store)
    graded_mode = any(t.get("reference") for t in tasks)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for strat in strategies:
            rcfg = copy.deepcopy(cfg)
            if task.get("rubric"):
                rcfg.setdefault("judge", {})["rubric"] = task["rubric"]
            t0 = time.time()
            sess = orchestrator.run_session(rcfg, task["task"], store=store,
                                            strategy=strat, prov=prov, verbose=False)
            secs = time.time() - t0

            errored = (sess.status != "ok") or not (sess.final or "").strip()
            match_score, correct, g_cost, g_tokens = None, None, 0.0, 0
            if task.get("reference") and not errored:
                match_score, correct, gturn = grade.grade(
                    cfg, prov, task["task"], sess.final, task["reference"], store=store,
                    match=task.get("match"))
                if gturn:
                    g_cost, g_tokens = gturn.cost_usd, gturn.tokens_in + gturn.tokens_out

            row = {
                "strategy": strat, "task_id": task["id"], "score": round(sess.final_score, 2),
                "rounds": len([r for r in sess.rounds if r.index != 0]),
                "tokens_in": sess.tokens_in, "tokens_out": sess.tokens_out,
                "tokens": sess.tokens_in + sess.tokens_out + g_tokens,
                "cost_usd": round(sess.cost_usd + g_cost, 6), "seconds": round(secs, 3),
                "match": None if match_score is None else round(match_score, 2),
                "correct": correct, "error": errored,
            }
            rows.append(row)
            store.add_bench_row(strat, task["id"], row["score"], row["rounds"],
                                row["tokens_in"], row["tokens_out"], row["cost_usd"], row["seconds"],
                                match=row["match"], correct=row["correct"])
            if verbose:
                if errored and task.get("reference"):
                    extra = "  [ERR: no answer / throttled]"
                elif row["match"] is not None:
                    mark = "OK" if correct else ("X" if correct is False else "?")
                    extra = f"  match={row['match']:>5.1f} [{mark}]"
                else:
                    extra = ""
                print(f"  {task['id']:<14} {strat:<9} score={row['score']:>5.1f}  "
                      f"rounds={row['rounds']}  tokens={row['tokens']:>5}  "
                      f"cost=${row['cost_usd']:.4f}  {row['seconds']:.2f}s{extra}")

    summary = aggregate(rows, strategies, len(tasks))
    if as_json:
        print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    else:
        print("\n" + _table(summary, graded=graded_mode))
    store.add_run("bench", len(rows), "ok")
    return 0


def aggregate(rows: list[dict[str, Any]], strategies: list[str], n_tasks: int) -> list[dict[str, Any]]:
    # Rank metric: match-vs-reference when graded, else the rubric score.
    def metric(r: dict[str, Any]) -> float:
        return r["match"] if r.get("match") is not None else r["score"]

    # wins: per task, the strategy with the best metric (ties -> shared).
    wins: dict[str, float] = {s: 0.0 for s in strategies}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    for task_rows in by_task.values():
        best = max(metric(r) for r in task_rows)
        leaders = [r for r in task_rows if metric(r) == best]
        for r in leaders:
            wins[r["strategy"]] += 1.0 / len(leaders)

    out = []
    for s in strategies:
        sr = [r for r in rows if r["strategy"] == s]
        if not sr:
            continue
        k = len(sr)
        entry = {
            "strategy": s,
            "mean_score": round(sum(r["score"] for r in sr) / k, 2),
            "win_rate": round(100.0 * wins[s] / max(1, n_tasks), 1),
            "mean_rounds": round(sum(r["rounds"] for r in sr) / k, 2),
            "mean_tokens": int(sum(r["tokens"] for r in sr) / k),
            "mean_cost_usd": round(sum(r["cost_usd"] for r in sr) / k, 6),
            "mean_seconds": round(sum(r["seconds"] for r in sr) / k, 3),
        }
        graded = [r for r in sr if r.get("match") is not None]
        if graded:
            entry["mean_match"] = round(sum(r["match"] for r in graded) / len(graded), 2)
            flags = [r["correct"] for r in graded if r.get("correct") is not None]
            entry["accuracy"] = round(100.0 * sum(1 for f in flags if f) / len(flags), 1) if flags else None
        entry["errors"] = sum(1 for r in sr if r.get("error"))
        entry["served"] = len(graded)
        out.append(entry)

    rank_key = "mean_match" if any("mean_match" in e for e in out) else "mean_score"
    out.sort(key=lambda r: (r.get(rank_key, 0) or 0, r["win_rate"]), reverse=True)
    return out


def _table(summary: list[dict[str, Any]], graded: bool = False) -> str:
    if graded:
        head = (f"{'strategy':<10} {'match':>6} {'acc%':>6} {'err':>4} {'score':>6} {'win%':>6} "
                f"{'rounds':>7} {'tokens':>7} {'cost$':>9} {'sec':>6}")
        lines = [head, "-" * len(head)]
        for r in summary:
            acc = r.get("accuracy")
            acc_s = f"{acc:>6.1f}" if acc is not None else f"{'-':>6}"
            lines.append(f"{r['strategy']:<10} {r.get('mean_match', 0):>6.1f} {acc_s} "
                         f"{r.get('errors', 0):>4} {r['mean_score']:>6.1f} {r['win_rate']:>6.1f} "
                         f"{r['mean_rounds']:>7.2f} {r['mean_tokens']:>7} {r['mean_cost_usd']:>9.4f} "
                         f"{r['mean_seconds']:>6.2f}")
        if summary:
            top = summary[0]
            acc = top.get("accuracy")
            note = f", accuracy {acc:.0f}% over {top.get('served', 0)} served" if acc is not None else ""
            errs = f"; {top.get('errors', 0)} errored/throttled" if top.get("errors") else ""
            lines.append(f"\n  winner: {top['strategy']} (match {top.get('mean_match', 0):.1f}{note}{errs})")
        return "\n".join(lines)

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


def _resolve_tasks_path(path: str) -> str:
    """Allow a bare eval name (``--tasks reasoning``) to resolve to a shipped set
    under ``evals/``; otherwise use the path as given."""
    if os.path.exists(path):
        return path
    if not os.path.splitext(path)[1] and "/" not in path and os.sep not in path:
        for ext in (".yaml", ".yml", ".json"):
            cand = os.path.join("evals", path + ext)
            if os.path.exists(cand):
                return cand
    return path


def _load_tasks(path: str) -> list[dict[str, Any]]:
    path = _resolve_tasks_path(path)
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
                          "rubric": item.get("rubric"),
                          "reference": item.get("reference") or item.get("expected") or item.get("answer"),
                          "match": item.get("match")})
    return tasks
