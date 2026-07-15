# Testing & Verification Schedule — quorum

The engine is deterministic and **fully offline** via the `mock` provider, so the
core is verified with zero network and zero keys. The **live** path (real models)
is a separate, key-gated, $0-on-`:free` check. This schedule says *what* to run,
*when*, and the *pass bar*.

## What "verified" means
1. **Offline engine green** — `selftest` + `pytest` pass with no network.
2. **Live reachable** — at least one real model answers per role (`models --ping`).
3. **Guardrails intact** — no secrets committed; every model `:free` (or priced); peer text framed as data.

## Cadence

| Trigger | Scope | Command(s) | Pass criteria | ~Time |
|---|---|---|---|---|
| Every code change | quick sanity | `python -m quorum selftest` | all `PASS`, exit 0 | ~1s |
| Dataset/pack change | pack integrity | `python -m quorum packs verify` | all six shipped packs pass | ~1s |
| Before each commit | full offline | `python -m quorum selftest` **and** `pytest -q` | 0 failures both | ~30s |
| After provider/config edits | live reachability | `quorum models --ping` | ≥1 `[ok]` per role | 10–40s |
| Before a release / version bump | full matrix | offline + live smoke + bench + dashboard | see checklist below | ~5 min |
| Weekly | free-model drift | compare config IDs vs OpenRouter `/models`; `quorum models --ping` | IDs still offered + reachable | ~2 min |
| Monthly | deps + security | upgrade deps, re-run suite; re-read injection framing | suite green; framing intact | ~10 min |
| On demand (debug) | one strategy | `quorum run "<task>" --strategy X --json` | completes, final non-empty, score > 0 | varies |

## Layers

### 1. Offline — must always pass (no key, no network)
- `python -m quorum selftest` — executable checks across config, model, store, cost, provider(mock), judge + stop logic, all strategies, promptsmith, bench, render, and export. Current baseline: 144 passed, 0 failed.
- `pytest -q` — unit/e2e coverage, including source locks, pack fingerprints, split leakage, and additive database migration. Current baseline: 231 passed.
- `python -m quorum packs verify` — validates all six shipped smoke packs, licenses/provenance, split fingerprints, task IDs, graders, and exact cross-split leakage.
- `quorum packs fetch` is deliberately excluded from CI: it is the explicit network step that writes raw sources and a mutable-source lock under ignored `data/`.
- Determinism anchors: mock judge ramps `55 + 15·round`; `debate`/`council`/`refine` stop at round 2; `moa` = 96; `ensemble` = 70.

### 2. Live smoke — key-gated, $0 on `:free`
- `quorum models --ping` — endpoint + auth reachability.
- `quorum run "<task>" --strategy refine --no-promptsmith` — cheapest complete path (~4 calls); best first live check.
- Escalate to `debate` → `council` → `moa` as rate limits allow.
- Free-tier notes: keep `run.parallel: false`; retry/backoff absorbs 429s; a heavy multi-model debate may need a little OpenRouter credit to avoid throttling.

### 3. Drift & regression — periodic
- Free model IDs churn: confirm each configured model still appears in OpenRouter's `/models` list.
- Re-run the full offline suite after **any** dependency bump.

### 4. Cost & security — every release
- Every configured model ends in `:free`, **or** has a `cost.pricing` entry.
- `cost.budget_usd` guard aborts over-budget runs (selftest covers `over_budget`).
- Prompt-injection (OWASP LLM01): peer/candidate text stays framed as DATA — grep the `QUORUM-` sentinels in judge/review/chairman/aggregator system prompts.
- Secrets: `.env` is gitignored; `git ls-files | Select-String env` shows only `.env.example`.

## Pre-release checklist (v0.x.y)
- [ ] `python -m quorum selftest` → `0 failed`
- [ ] `pytest -q` → all pass
- [ ] `python -m quorum packs verify` → all six packs pass
- [ ] `quorum models --ping` → each role reachable (needs a key)
- [ ] one live `quorum run` returns a non-empty final answer
- [ ] `quorum bench --tasks tests/fixtures/tasks.small.yaml --strategies debate,council,moa,refine,ensemble` → prints table + winner
- [ ] `quorum dashboard` builds; open it and spot-check one transcript
- [ ] `git ls-files` contains no `.env`, `data/`, or `config.yaml`
- [ ] version bumped in `pyproject.toml` **and** `quorum/__init__.py`
- [ ] README commands still accurate

## CI (offline, no key)
GitHub Actions on push/PR:
```yaml
- run: pip install -e ".[dev]"
- run: python -m quorum selftest
- run: pytest -q
```
Everything runs on the `mock` provider, so CI needs no API key.
