# quorum

**Put several AI models in a room, refine a prompt together, then debate a solution until it's good enough.**

`quorum` is a small, provider-agnostic CLI that runs a *deliberation* between multiple language models. It first **designs and refines the prompt** (phase 1), then bounces a solution between the models — proposing, critiquing, and revising — while a **judge scores every round** and stops when the answer crosses a quality bar (or plateaus, or hits a round cap). Three deliberation strategies ship in the box, plus two cheap baselines, and a **benchmark harness** tells you which one actually wins on *your* tasks.

The whole engine — rounds, judging, stopping, storage, rendering, cost accounting, the benchmark — is deterministic and **runs fully offline** via a built-in `mock` provider (that's what `selftest` and the test suite use). Live deliberation talks to any OpenAI-compatible endpoint; keys come from the environment and nothing leaves your machine except the model calls you configure.

---

## Why (the research)

Multiple-model collaboration is well studied, and `quorum` bakes the lessons in rather than reinventing them:

| Idea | Paper | How quorum uses it |
| --- | --- | --- |
| Multi-agent debate | Du et al. 2023 ([2305.14325](https://arxiv.org/abs/2305.14325)); Liang et al. MAD, EMNLP'24 ([2305.19118](https://arxiv.org/abs/2305.19118)) | `debate` strategy + a **judge** with **adaptive stopping** |
| Mixture-of-Agents | Wang et al. 2024 ([2406.04692](https://arxiv.org/abs/2406.04692)) | `moa` strategy (layered proposers + aggregator) |
| Council + chairman | Karpathy `llm-council` (2025) | `council` strategy, extended to **iterate** instead of single-pass |
| Self-Refine | Madaan et al. 2023 ([2303.17651](https://arxiv.org/abs/2303.17651)) | `refine` baseline |
| Prompt optimization (OPRO) | Yang et al. 2023 ([2309.03409](https://arxiv.org/abs/2309.03409)) | `promptsmith` phase-1 prompt refinement |
| *"Debate isn't always worth it"* | Smit et al. 2023 ([2311.17371](https://arxiv.org/abs/2311.17371)) | cheap `ensemble` baseline + the `bench` harness to prove which wins |

That last one matters: multi-agent debate does **not** reliably beat cheaper methods and is hyperparameter-sensitive. So `quorum` keeps the knobs exposed, keeps a cheap ensemble baseline in the ring, and lets you benchmark rather than assume.

## Install

```powershell
git clone <your-fork> quorum
cd quorum
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Requires Python 3.11+. The only runtime dependency is PyYAML. `pip install -e ".[tokens]"` adds `tiktoken` for exact token counts (optional); `".[dev]"` adds pytest.

## Quickstart

```powershell
quorum init                        # scaffold config.yaml + data/ + .env
quorum selftest                    # offline sanity check — no key needed
```

Set one key (OpenRouter unlocks OpenAI/Anthropic/Google/xAI/Meta with a single credential):

```
# .env
QUORUM_OPENROUTER_KEY=sk-or-v1-...
```

Then deliberate:

```powershell
quorum models --ping                                   # confirm endpoints are reachable
quorum run "Design a rate limiter for a public API." --strategy council
quorum run "What's the capital of Australia, and the common wrong guess?"
```

No key? Everything still runs offline against the mock provider:

```powershell
quorum --config config.mock.yaml run "anything" --strategy debate
```

## Strategies

| Name | What happens | Cost |
| --- | --- | --- |
| `debate` | Every member answers, then each revises after seeing the others' (anonymized) answers + the judge's critique. Judge scores the best each round. | medium |
| `council` | Members answer → peer-review anonymized answers → a **chairman** synthesizes → judge scores → repeat if not good enough. | higher |
| `moa` | Layered mixture-of-agents: each layer sees the previous layer's outputs; an **aggregator** merges the final layer. | medium |
| `refine` | Single model generates → critiques itself → revises. A cheap single-agent baseline. | low |
| `ensemble` | Sample one model N times; the judge picks the best. Self-consistency baseline. | low |

Pick a default in `config.yaml` (`run.strategy`) or override with `--strategy`.

## Which strategy is best? Benchmark it.

```powershell
quorum bench --tasks tests/fixtures/tasks.small.yaml --strategies debate,council,moa,refine,ensemble
```

```
strategy    score   win%  rounds  tokens     cost$    sec
---------------------------------------------------------
moa          96.0  100.0    3.00    1927    0.0000   0.03
debate       85.0    0.0    2.00    2199    0.0000   0.06
council      85.0    0.0    2.00    3520    0.0000   0.07
refine       85.0    0.0    2.00    1234    0.0000   0.02
ensemble     70.0    0.0    1.00    1082    0.0000   0.02

  winner: moa (mean score 96.0, win-rate 100%)
```

(Numbers above are the deterministic mock; real models will differ — that's the point of running it on your own tasks.)

## "Good enough" — the stop rule

A round ends the deliberation when **any** of these is true:

- the best judge score reaches `run.target_score` (default 85), or
- the score plateaus (< `run.plateau_delta` gain for `run.plateau_patience` rounds), or
- `run.max_rounds` is reached (guarantees termination), or
- `run.consensus: true` and the members converge on one answer, or
- the projected spend exceeds `cost.budget_usd`.

## Providers (free + local friendly)

Any OpenAI-compatible `/chat/completions` endpoint works. Configure a roster of any size:

```yaml
council:
  members:
    - { name: alice, provider: openrouter, model: meta-llama/llama-3.1-8b-instruct:free }
    - { name: bob,   provider: ollama,     model: llama3.1 }          # local, keyless
    - { name: carol, provider: openai,     model: gpt-4o-mini }
  judge:    openrouter:meta-llama/llama-3.3-70b-instruct:free
  chairman: openrouter:meta-llama/llama-3.3-70b-instruct:free
providers:
  openrouter: { base_url: https://openrouter.ai/api/v1, api_key_env: QUORUM_OPENROUTER_KEY }
  ollama:     { base_url: http://localhost:11434/v1,    api_key_env: "" }
```

Secrets never live in config — each provider names an environment variable.

## Commands

| Command | Purpose |
| --- | --- |
| `quorum init` | Scaffold config + data dir + `.env` |
| `quorum run "<task>"` | Deliberate to a good-enough solution (`--strategy --rounds --target --no-promptsmith --json`) |
| `quorum promptsmith "<task>"` | Just design/refine a prompt (phase 1) |
| `quorum bench --tasks f --strategies a,b,c` | Compare strategies over a task set |
| `quorum list` / `quorum show <id>` | Browse past deliberations |
| `quorum dashboard [--open]` | Build the offline HTML transcript browser |
| `quorum serve [--open]` | Serve the dashboard on `127.0.0.1:8802` |
| `quorum export [--format json\|csv\|md] [--session id]` | Export a transcript |
| `quorum models [--ping]` | List the council (and check reachability) |
| `quorum selftest` | Offline self-tests (no network, no keys) |

## Extending it

Strategies are plugins. Register your own under the `quorum.strategies` entry-point group (a callable `run(ctx) -> Session`) and it appears automatically in `run` and `bench`.

## Security

Model outputs are fed back into other models' prompts during deliberation, which is a prompt-injection surface (OWASP LLM01). `quorum` always frames peer text as **data to evaluate, not instructions to follow** in the judge/chairman/aggregator system prompts, and the provider never executes anything a model returns. Peer identities are anonymized during review by default (`run.anonymize`).

## Testing

```powershell
quorum selftest        # fast, offline, deterministic (mock provider)
pytest -q              # unit + end-to-end tests, all offline
```

## License

MIT
