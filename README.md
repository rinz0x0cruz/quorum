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

**Default: `refine`** — in a GSM8K reference eval it matched `debate`'s accuracy at roughly a third of the cost/latency, echoing the "Should we be going MAD?" finding that debate isn't always worth it. Switch to `debate`/`council`/`moa` for hard or contested prompts via `run.strategy` in `config.yaml` or `--strategy`.

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
Point any OpenAI client's `base_url` at `http://127.0.0.1:8802/v1` and pass a strategy name as the `model` (`refine`, `debate`, `council`, `moa`, `ensemble`). `stream: true` and a per-request `--timeout` are supported.

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
