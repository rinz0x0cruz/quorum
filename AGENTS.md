# AGENTS.md — quorum

Operating guide for AI coding agents working in this repo. Read this **before** editing.
Pair it with [`ARCHITECTURE.md`](ARCHITECTURE.md) (the authoritative code map + modularity roadmap).

## What this is

quorum turns a prompt into a *deliberated* answer — several models propose, critique, and
refine until a judge says "good enough" — and is also the shared AI backend for the sibling
tools (claudebudget, jobscope, learnscope, exploitrank). Pure Python. The **only** runtime
dependency is PyYAML; HTTP is stdlib `urllib`; concurrency is `concurrent.futures`. A built-in
`mock` provider answers deterministically so the whole engine runs offline with no keys.

## Golden rules (do not violate)

1. **Behavior-preserving refactors only.** Unless a task explicitly asks for a behavior change,
   the CLI surface, the embed API (`api.chat` / `api.deliberate` signatures), the OpenAI-compatible
   `serveapi` contract, and deliberation outputs on the `mock` provider must be **identical**
   before/after. Verify with the ship loop below.
2. **AI-optional & offline-testable.** The `mock` provider must exercise every code path; `selftest`
   and most of pytest run with **no network and no keys**. Never make the deterministic core depend
   on a live model. When quorum is embedded in a host tool, it stays an *optional* layer.
3. **Lean dependencies.** Runtime dep stays **PyYAML only**. Optional extras (e.g. `tiktoken`) must
   be import-guarded and never gate core behavior. **Do not add a new runtime dependency.**
4. **Respect the layers** (see `ARCHITECTURE.md §3`). Arrows point down: reasoning services
   (`provider`, `judge`, `prompts`, `rank`, `contextwindow`, `promptsmith`, `grade`) must not import
   strategies; the domain layer (`model`, `config`, `cost`, `store`) imports nothing above it. No
   import cycles.
5. **Feature flags default-off.** New knobs live in `DEFAULT_CONFIG` + `config.example.yaml` and
   default to the pre-feature behavior. Keep `api.chat` **signature-compatible** so the sibling
   tools' delegation keeps working.
6. **Every change ships with tests.** Add a `selftest` check **and** a pytest, both offline and
   deterministic, for anything you add or move.
7. **Don't `git push` unless explicitly told.** Commit locally with small, scoped, imperative
   messages; leave pushing to the human/orchestrator. Do not change git author config.

## Ship loop (run after every change; all must be green)

On this Windows host, PowerShell blocks the venv `Activate.ps1` — call the venv Python directly
(do **not** try to activate):

```powershell
Set-Location 'C:\Users\Blanc\Git\quorum'
.\.venv\Scripts\python.exe -m pytest -q                 # all pass
.\.venv\Scripts\python.exe -m quorum selftest           # all checks pass (0 failed)
```

Also confirm there are no lint/compile errors in the files you touched. A refactor is "green"
only when the pytest count is **>=** the pre-change count (existing tests unchanged) and selftest
reports `0 failed`.

## Conventions

- **Style:** `from __future__ import annotations`; type hints on public functions; module- and
  function-level docstrings. Feature modules are lazy-imported in `__main__.py` to keep startup light.
- **Registries over switches.** New strategies register via the `quorum.strategies` entry-point
  group; new pipeline stages attach via `hooks.register_pre/post` (`quorum.hooks.pre/post`). Prefer
  this pattern to editing a central switch.
- **Config:** one `DEFAULT_CONFIG` deep-merged with the user's file; secrets come from env vars named
  by each provider profile. Mirror the sibling tools' config system.
- **`RunOptions`:** run-level knobs are resolved once in the orchestrator and read via `ctx.opts.*`
  inside strategies — add a new run knob to `RunOptions` + `from_cfg`, not to each strategy.
- **Security (OWASP LLM01):** any model output or caller-supplied text (peer answers, history,
  context docs) fed back into a prompt is framed as **DATA, never instructions**. Preserve that
  framing when you move prompt code.
- **Commits:** small, imperative, scoped — e.g. `refactor(scoring): unify judge/grade/rank behind a
  scorer protocol`. **Do not push.**

## Environment gotchas

- Fresh PowerShell shells have a stale machine `PATH`; if you need `git`/`gh`, prefix with:
  `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`.
- `git push` writes progress to stderr, which PowerShell surfaces as a red `NativeCommandError`
  even on success — check the `->` ref-update line, not the exit stream. (You should not be pushing anyway.)
- Call `.\.venv\Scripts\python.exe` directly; the execution policy blocks `Activate.ps1`.

## Where things are (see `ARCHITECTURE.md` for the full map)

- `quorum/` — the package. Entry points: `__main__.py` (CLI), `api.py` (embed), `serveapi.py`
  (OpenAI-compatible proxy). Orchestration: `orchestrator.py`, `strategies/` (registry + `Context`
  + `RunOptions`), `hooks.py`. Reasoning: `provider`, `judge`, `prompts`, `promptsmith`, `rank`,
  `contextwindow`, `grade`. Domain: `model`, `config`, `store`, `cost`. Reporting: `bench`, `render`,
  `format`, `exporter`, `serve`, `scaffold`, `selftest`.
- `tests/` — pytest suite; `quorum/selftest.py` — the offline checks. The `mock` provider is the
  contract that keeps everything offline-verifiable.
- The **modularity roadmap** (items #3 scoring package, #4 adapters, #5 prompts, …) lives in
  `ARCHITECTURE.md §8`. When you complete an item, update that table to mark it done.
