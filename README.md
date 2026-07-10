# quorum

[![ci](https://github.com/rinz0x0cruz/quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/rinz0x0cruz/quorum/actions/workflows/ci.yml)

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
| Self-consistency + USC | Wang et al. 2022; Chen et al. 2023 ([2311.17311](https://arxiv.org/abs/2311.17311)) | `selfconsistency` strategy — majority vote or USC selection for free-form |
| Reflexion | Shinn et al. 2023 ([2303.11366](https://arxiv.org/abs/2303.11366)) | `reflexion` strategy — verbal self-reflection kept in memory across attempts |
| Chain-of-Verification | Dhuliawala et al. 2023 ([2309.11495](https://arxiv.org/abs/2309.11495)) | `verify` strategy — independent verification questions before a final answer |
| Self-MoA (is mixing worth it?) | Li et al. 2025 ([2502.00674](https://arxiv.org/abs/2502.00674)) | `selfmoa` strategy — sample the single best model + aggregate; `bench` it vs `moa`/`council` |
| Self-Discover | Zhou et al. 2024 ([2402.03620](https://arxiv.org/abs/2402.03620)) | `selfdiscover` strategy — compose a task-specific reasoning structure, then solve it (cheap: ~2 calls) |
| Step-Back prompting | Zheng et al. 2023 ([2310.06117](https://arxiv.org/abs/2310.06117)) | `stepback` strategy — abstract to the governing principle first, then solve (cheap: ~2 calls) |
| Least-to-Most | Zhou et al. 2022 ([2205.10625](https://arxiv.org/abs/2205.10625)) | `leasttomost` strategy — decompose into ordered sub-questions, then solve them in sequence |
| Adaptive-Consistency | Aggarwal et al. EMNLP'23 ([2305.11860](https://arxiv.org/abs/2305.11860)) | `run.adaptive_samples` — sample until a confident majority, then stop |
| LLM cascade (FrugalGPT) | Chen, Zaharia & Zou 2023 ([2305.05176](https://arxiv.org/abs/2305.05176)) | `cascade` strategy — cheap first, escalate only if the judge isn't satisfied |
| LLM-as-judge biases | Zheng et al. NeurIPS'23 ([2306.05685](https://arxiv.org/abs/2306.05685)) | judge **shuffles candidate order** to curb position bias |

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
| `selfconsistency` | Sample one model N times; return the **consensus** — majority vote (numeric) or USC selection (free-form). Optional adaptive early-stop. | low |
| `selfmoa` | Sample the **single strongest** model N times and aggregate them — often beats mixing weaker models on free tiers. | medium |
| `reflexion` | Single actor answers → judge scores → it writes a **self-reflection** kept in memory; each retry learns from all past reflections. | low |
| `verify` | **Chain-of-verification**: draft → plan checks → answer them *independently* (draft withheld) → revise into a verified final. Cuts hallucination. | medium |
| `cascade` | Run strategies cheapest-first (`refine`→`debate`→`council`) and stop at the first to clear the target — spends more **only on hard tasks**. | adaptive |
| `selfdiscover` | **Compose a reasoning structure**: select + adapt atomic reasoning modules into a task-specific plan, then follow it to solve. Structured reasoning in ~2 single-model calls. | low |
| `stepback` | **Step back to a principle**: first derive the general concept/rule behind the task, then reason from it to the answer. Curbs slips in ~2 calls. | low |
| `leasttomost` | **Decompose then chain**: break the task into ordered sub-questions and solve them in sequence, each answer feeding the next. Strong on compositional problems; solve calls capped. | medium |

**Default: `refine`** — in a GSM8K reference eval it matched `debate`'s accuracy at roughly a third of the cost/latency, echoing the "Should we be going MAD?" finding that debate isn't always worth it. Switch to `debate`/`council`/`moa` for hard or contested prompts via `run.strategy` in `config.yaml` or `--strategy`.

## Running on free tiers (fewer requests, less throttling)

Free model endpoints cap you by *request count*: OpenRouter's `:free` models allow ~20 requests/minute and 50/day (1000/day once you've purchased ≥10 credits), governed **globally** across keys. quorum has knobs to spend requests only when a task needs them:

- **`run.rate_limit_rpm`** — pace HTTP calls per provider (e.g. `18`) so parallel bursts don't trip the per-minute cap. The single biggest 429-killer.
- **`cascade` strategy** — solve easy tasks with one cheap model, escalate to debate/council only when the judge isn't satisfied.
- **`run.adaptive_samples`** — for `ensemble`/`selfconsistency`, stop sampling once a confident majority emerges (often far fewer than `run.samples`).
- **`run.judge_every`** — judge every N rounds instead of every round in `debate`/`council`/`refine`.
- On a 429, quorum honors `Retry-After` and **rotates to a fallback model** (separate limit) instead of hammering the throttled one.

See what's actually throttling you — per-model 429 rate, peak requests/min vs the ceiling, remaining daily quota, and concrete fixes:

```powershell
quorum throttle
```

It reads per-attempt telemetry recorded during live runs (status, latency, `Retry-After`, `X-RateLimit-*`) and probes your key's remaining quota. The offline dashboard (`quorum dashboard`) also shows an **api throttle** panel with the same per-model 429 rate and peak requests/min.

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

### Grade on correctness, not just judge score

Give a task a gold `answer` and a `match` type, and `bench` scores every strategy on **accuracy** — deterministically, with **no grader model** (free and objective):

```powershell
quorum bench --tasks reasoning --strategies refine,selfconsistency,verify,selfmoa,cascade
```

`--tasks reasoning` resolves to the shipped [`evals/reasoning.yaml`](evals/reasoning.yaml) (a small arithmetic / multiple-choice / yes-no / short-answer set); a harder companion set lives at [`evals/reasoning-hard.yaml`](evals/reasoning-hard.yaml) (`--tasks reasoning-hard` — System-1 traps + multi-step problems that separate strategies on accuracy, not just cost). `match` is one of:

| `match` | grades by | example gold |
|---|---|---|
| `numeric` *(auto)* | the final number (a bare number or a GSM8K `#### 42` line) | `#### 72` |
| `choice` | the A/B/C/… letter | `B` |
| `boolean` | yes/no · true/false | `yes` |
| `exact` | normalized final word/phrase | `Canberra` |
| `contains` | gold appears anywhere in the answer | `decrease` |
| `regex` | a pattern matches the answer | `\b42\b` |

Numeric is auto-detected, so existing GSM8K-style task files keep working. Prompts that end with `Answer: <x>` extract cleanly even after chain-of-thought; tasks with no `match` (and no gold number) fall back to the LLM grader. The graded run adds `match`/`acc%` columns and ranks by match instead of judge score:

```
strategy    match   acc%  err  score   win%  rounds  tokens  cost$   sec
-----------------------------------------------------------------------
verify       ..     ..     0    ..     ..     ..      ..    0.0000   ..
selfmoa      ..     ..     0    ..     ..     ..      ..    0.0000   ..
...
```

(Grading is free — `cost$` stays `0.0000` — so the only spend is the deliberation itself. Run it live on your models for a real correctness ranking.)

## Scoring model

Every round, an impartial **judge** model scores the best candidate on a **0–100** scale
against a weighted rubric (weights are normalized, so they need not sum to 1). The defaults:

| Criterion | Weight | Rewards |
| --- | --- | --- |
| `correctness` | 0.40 | factually right, sound reasoning, actually does the task |
| `completeness` | 0.25 | covers every part of the ask, no gaps |
| `clarity` | 0.20 | well-structured, unambiguous, easy to follow |
| `grounding` | 0.15 | claims supported by the given context/sources, not invented |

Tune the weights (or add criteria) under `judge.rubric` in `config.yaml`. The judge runs at
`temperature: 0` for stable scores, treats every candidate as **data to evaluate, not
instructions** (OWASP LLM01 mitigation), and by default is drawn from a different model
family than the candidate (`judge.cross_family_guard`) to curb self-preference bias. That
score is what drives the stop rule below.

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
    - { name: alice, provider: openrouter, model: meta-llama/llama-3.3-70b-instruct:free }
    - { name: bob,   provider: ollama,     model: llama3.1 }          # local, keyless
    - { name: carol, provider: openai,     model: gpt-4o-mini }
  judge:    openrouter:openai/gpt-oss-120b:free
  chairman: openrouter:openai/gpt-oss-120b:free
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
| `quorum serve --api [--host --port --token --timeout]` | OpenAI-compatible `/v1/chat/completions` endpoint (deliberates per request) |
| `quorum chat --system "…" --user "…" [--json --strategy X]` | One-shot deliberation for scripts / CI / other languages |
| `quorum export [--format json\|csv\|md] [--session id]` | Export a transcript |
| `quorum models [--ping]` | List the council (and check reachability) |
| `quorum selftest` | Offline self-tests (no network, no keys) |

## Extending it

Strategies are plugins. Register your own under the `quorum.strategies` entry-point group (a callable `run(ctx) -> Session`) and it appears automatically in `run` and `bench`.

## Embed it in another tool (replace a single generic AI call)

`quorum` is also a library. Any tool that currently makes one generic model call can route that call through a deliberation instead — self-refine by default, or a full council — while keeping AI **optional**.

Install it into the host tool's environment:

```powershell
pip install -e path\to\quorum
```

Add a `quorum:` block to the host's existing `config.yaml` (its `ai:` block supplies the default provider/model/key):

```yaml
quorum:
  enabled: true          # off by default -> AI stays optional
  strategy: refine        # refine | debate | council | moa | ensemble
  max_rounds: 2
  # Optional extra council members; omit to just self-refine the tool's own ai.model:
  members:
    - { name: a, provider: openrouter, model: google/gemma-4-31b-it:free }
    - { name: b, provider: openrouter, model: openai/gpt-oss-120b:free }
  providers:
    openrouter: { base_url: https://openrouter.ai/api/v1, api_key_env: TOOL_OPENROUTER_KEY }
```

Then delegate at the top of the host's `ai.py::chat` — a 6-line change that preserves the existing single-model fallback:

```python
def chat(cfg, store, system, user, *, temperature=None):
    try:
        from quorum.api import chat as _q
        out = _q(cfg, store, system, user, temperature=temperature)
        if out is not None:
            return out
    except ImportError:
        pass
    # ... existing single-model path, unchanged ...
```

- **Optional & safe**: if `quorum` isn't installed, `quorum.enabled` is false, or no key is set, `quorum.api.chat` returns `None` and the host falls back to its own behavior — so the tool still runs with zero AI.
- **Signature-compatible** with the sibling tools' `ai.chat(cfg, store, system, user)`; the host's `system` prompt drives the deliberation (promptsmith is skipped).
- **Store-friendly**: pass the host's store — quorum reuses its `ai_cache` if present and never requires the session tables.
- **Per-call strategy**: pass `strategy=` to `api.chat` to choose the deliberation for *one* call — `council`/`debate` for a high-stakes answer, `ensemble` for a discrete classification — overriding the host's configured `quorum.strategy`.
- **Second-opinion scoring**: `api.score(cfg, store, task, candidate, *, rubric=None)` runs quorum's rubric judge over a single answer and returns `{"score", "sub_scores", "rationale"}` (or `None` when off) — a tool with its own deterministic ranker can use it as an optional cross-check without touching its core scoring.

## Use it from any language / environment

Beyond the Python library, quorum exposes two language-agnostic surfaces.

### OpenAI-compatible proxy (any language, incl. Go/Rust)
Run a local server that deliberates per request and answers in the OpenAI shape:
```powershell
quorum serve --api --port 8802            # add --token <secret> to require auth; --host 0.0.0.0 to expose
```
Point any OpenAI client's `base_url` at `http://127.0.0.1:8802/v1` and pass a strategy name as the `model` (`refine`, `debate`, `council`, `moa`, `ensemble`, `selfconsistency`, `selfmoa`, `reflexion`, `verify`, `cascade`, `selfdiscover`, `stepback`, `leasttomost`). With `stream: true` the response is server-sent events — **live progress** (round scores, stop reason) as SSE comments during the deliberation, then the final answer as an OpenAI `chat.completion.chunk`. A per-request `--timeout` is supported.

**Docker** (zero local Python):
```bash
docker build -t quorum .
docker run --rm -p 8802:8802 -e QUORUM_OPENROUTER_KEY=sk-or-... \
  -v "$PWD/config.yaml:/app/config.yaml:ro" quorum
```
No registry needed — CI builds the image and uploads it as a `quorum-docker-image` artifact; download it from the latest [Actions run](https://github.com/rinz0x0cruz/quorum/actions) and `docker load < quorum-image.tar.gz`.

**Rust** (through the proxy, e.g. with `reqwest` or any OpenAI client):
```rust
let body = serde_json::json!({
    "model": "refine",
    "messages": [{"role":"system","content":"be terse"},
                 {"role":"user","content":"summarize this CVE"}],
});
let r: serde_json::Value = reqwest::Client::new()
    .post("http://127.0.0.1:8802/v1/chat/completions")
    .json(&body).send().await?.json().await?;
let answer = &r["choices"][0]["message"]["content"];
```

### Subprocess one-shot (CI, no daemon)
For unattended/CI use where running a server is awkward, spawn a one-shot — stdout is the answer:
```bash
quorum chat --system "be terse" --user "summarize this CVE" --strategy refine
echo "summarize this CVE" | quorum chat --system "be terse" --json   # {content, tokens, cost_usd, ...}
```
Any language can `exec` it (Rust `std::process::Command`, Go `os/exec`). The exit code is non-zero when no answer is produced, so callers can fall back.

### Install pinned to a version
```bash
pip install "quorum @ git+https://github.com/rinz0x0cruz/quorum@v0.1.0"
```

## Security


Model outputs are fed back into other models' prompts during deliberation, which is a prompt-injection surface (OWASP LLM01). `quorum` always frames peer text as **data to evaluate, not instructions to follow** in the judge/chairman/aggregator system prompts, and the provider never executes anything a model returns. Peer identities are anonymized during review by default (`run.anonymize`).

## Testing

```powershell
quorum selftest        # fast, offline, deterministic (mock provider)
pytest -q              # unit + end-to-end tests, all offline
```

## License

MIT
