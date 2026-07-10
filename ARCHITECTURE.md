# quorum — architecture & code map

A living map of the codebase: what each module does, how a deliberation flows,
where the extension points are, and how the design should evolve as features are
added. Keep this current when you add or move a module.

---

## 1. What quorum is (and the rules it lives by)

quorum turns a prompt into a *deliberated* answer: several models propose,
critique, and refine until a judge says "good enough". It is also the shared AI
backend for the sibling tools (claudebudget, jobscope, learnscope, exploitrank).

Four principles constrain every design decision here:

- **Stateless engine.** quorum holds no cross-call memory. Callers own state
  (conversation history, corpora) and pass it in per call. This keeps the engine
  pure and trivially testable.
- **AI-optional everywhere it embeds.** When quorum is the backend for another
  tool, it is an *optional* enrichment layer — off by default, and the host
  degrades to its own deterministic path if quorum is absent/disabled.
- **Lean dependencies.** Runtime dep is PyYAML only; HTTP is stdlib `urllib`;
  concurrency is `concurrent.futures`. Optional extras (`tiktoken`) never gate
  core behavior.
- **Offline-testable.** A built-in `mock` provider answers deterministically, so
  the whole engine — every strategy, the judge, retrieval, the API — runs with no
  network and no keys. `selftest` and most of pytest rely on it.

> Adding anything? It must keep all four true: default-off, mock-supported,
> no new hard dep, and covered by a selftest check + a pytest.

---

## 2. Module map

Grouped by layer (all under `quorum/quorum/`).

### Entry points — turn an external request into a `run_session`
| Module | Responsibility | Key symbols |
|---|---|---|
| `__main__.py` | CLI (argparse); lazy-imports feature modules | `cmd_run/chat/bench/serve/...` |
| `api.py` | **Embed API** — drop-in for a host tool's `ai.chat` | `enabled`, `build_config`, `deliberate`, `chat` |
| `serveapi.py` | **OpenAI-compatible HTTP server** (`serve --api`) for non-Python hosts | `complete_chat`, `make_server`, `run`, `_split` |
| `adapters.py` | **External -> quorum mappers** shared by `api` + `serveapi` (entry-layer helper) | `host_config`, `split_messages`, `select_strategy` |

### Orchestration — run one deliberation end to end
| Module | Responsibility | Key symbols |
|---|---|---|
| `orchestrator.py` | The pipeline: promptsmith → context preamble → pre-hooks → strategy → post-hooks → persist | `run_session(...)` |
| `strategies/__init__.py` | Strategy **registry** + entry-point discovery + shared `Context` + resolved `RunOptions` | `Context`, `RunOptions`, `get`, `available` |
| `hooks.py` | Pre/post extension hooks around the strategy (retrieval, filters, post-processing) | `register_pre`, `register_post`, `run_pre/post` |
| `strategies/{debate,council,moa,refine,ensemble,selfconsistency,selfmoa,reflexion,verify,cascade,selfdiscover,stepback,leasttomost}.py` | The deliberation algorithms | each exposes `run(ctx)` |

### Reasoning services — the moving parts a strategy composes
| Module | Responsibility | Key symbols |
|---|---|---|
| `provider.py` | All model I/O: mock-or-HTTP, retry/backoff, **fallbacks**, a per-provider **rate limiter**, per-attempt **throttle telemetry**, `response_format`, fan-out | `Provider`, `Completion`, `MockResponder`, `RateLimiter` |
| `judge.py` | Score a round vs a rubric; decide when to stop | `evaluate`, `should_stop`, `consensus_reached` |
| `prompts/` | System prompts + message builders (framed DATA-not-instructions); a **package** split by concern -- shared framing + generic builders in `base`, strategy-specific builders in `debate`/`council`/`moa`; every name re-exported via `__init__` | `propose`, `revise`, `challenge`, `synthesize`, `aggregate`, ... |
| `promptsmith.py` | Phase-1 OPRO prompt refinement + few-shot bootstrap | `refine`, `_exemplars` |
| `rank.py` | Rank candidates from peer reviews (Borda over reviewer orderings) | `consensus_order`, `top_k_indices` |
| `contextwindow.py` | Pack caller history + grounding docs into a DATA-framed preamble | `ContextDoc`, `pack`, `select`, `preamble` |
| `grade.py` | Reference grading: deterministic (numeric/choice/boolean/exact/contains/regex, no model) or AI grader fallback | `deterministic_match`, `numeric_match`, `final_answer`, `grade` |
| `scoring/` | Shared lexical text-scoring primitives + a `Scorer` protocol & registry (leaf) | `tokens`, `overlap_coeff`, `jaccard`, `LexicalScorer`, `register`/`get`/`available` |
| `consistency.py` | Cluster sampled answers (numeric-exact / lexical) + the adaptive-consistency stopping rule (leaf) | `assign`, `leader`, `confident`, `cluster` |
| `events.py` | Structured progress events + the `on_event` observability stream (leaf) | `Event`, `render`, `coerce` |

### Domain & data
| Module | Responsibility | Key symbols |
|---|---|---|
| `model.py` | Core dataclasses + id/vendor helpers | `ModelSpec`, `Turn`, `Verdict`, `Round`, `Session` |
| `config.py` | `DEFAULT_CONFIG`, deep-merge loader, `.env`, accessors | `load_config`, `member_specs`, `role_spec`, `api_key` |
| `store.py` | SQLite: sessions / ai_cache / bench / runs | `Store`, `save_session`, `ai_cache_*`, `top_sessions` |
| `cost.py` | Token counting (optional tiktoken) + pricing/budget | `count_tokens`, `price`, `over_budget` |

### Reporting & scaffolding
| Module | Responsibility |
|---|---|
| `bench.py` | Run/aggregate a strategy comparison over a task set |
| `render.py` | Self-contained offline HTML dashboard |
| `format.py` | Plain-text transcript for `run`/`show` |
| `exporter.py` | Export a session as JSON / CSV / Markdown |
| `serve.py` | Serve the dashboard over local HTTP |
| `throttle.py` | Analyze rate-limit telemetry (`quorum throttle`): per-model 429 rate, req/min vs the free ceiling, `/api/v1/key` quota probe, recommendations |
| `scaffold.py` | `init` — non-destructive config + data dir |
| `selftest.py` | ~90 offline checks; the extensibility contract in executable form |

---

## 3. Module dependencies

```mermaid
flowchart TD
    subgraph Entry
        CLI[__main__]
        API[api.py]
        SRV[serveapi.py]
    end
    subgraph Orchestration
        ORCH[orchestrator]
        REG[strategies/registry]
        STRAT[debate / council / moa / refine / ensemble]
    end
    subgraph Reasoning
        PROV[provider]
        JUDGE[judge]
        PROMPT[prompts]
        SMITH[promptsmith]
        RANK[rank]
        CTX[contextwindow]
        GRADE[grade]
    end
    subgraph Domain
        MODEL[model]
        CONFIG[config]
        STORE[store]
        COST[cost]
    end

    CLI --> ORCH
    API --> ORCH
    SRV --> ORCH
    API --> CONFIG
    SRV --> CTX
    ORCH --> REG --> STRAT
    ORCH --> SMITH
    ORCH --> CTX
    STRAT --> PROV
    STRAT --> JUDGE
    STRAT --> PROMPT
    STRAT --> RANK
    JUDGE --> PROV
    SMITH --> PROV
    PROV --> COST
    PROV --> STORE
    Reasoning --> MODEL
    Orchestration --> MODEL
    STRAT --> CONFIG
```

Rule of thumb: **arrows point down.** Reasoning services never import
strategies; the domain layer (`model`, `config`, `cost`, `store`) imports nothing
above it. Keep it that way — it is what makes the mock provider able to stand in
for the whole network boundary.

---

## 4. A deliberation, end to end

```mermaid
sequenceDiagram
    participant C as Caller (CLI / api / serveapi)
    participant O as orchestrator.run_session
    participant P as promptsmith
    participant X as contextwindow
    participant S as strategy.run(ctx)
    participant PR as provider
    participant J as judge
    participant DB as store

    C->>O: task (+ optional history/context, solve_prompt)
    O->>P: refine() unless solve_prompt given
    O->>X: preamble(history, context) → DATA block
    Note over O: prompt = preamble + solve/refined prompt
    O->>S: Context{cfg, prov, store, task, prompt, members, session, emit}
    loop until judge stops
        S->>PR: complete_many(proposals / revisions)
        S->>J: evaluate(candidates) → Verdict (+ optional rank.top_k)
        J-->>S: score, best, should_stop?
    end
    S-->>O: session.final, rounds, cost
    O->>DB: save_session (if a quorum store)
    O-->>C: Session
```

The **only** network boundary is `provider`. Swap in `MockResponder` and every
box above runs offline and deterministically.

---

## 5. Extension points (how to add things today)

- **A new strategy** — drop `strategies/yourstrat.py` exposing `run(ctx)`, add it
  to `_BUILTIN`, or ship it out-of-tree via the `quorum.strategies` entry-point
  group. The registry discovers installed plugins automatically.
- **A grounding/context source** — the caller (host tool) builds `ContextDoc`s
  and passes them to `api.chat(..., context=[...])` or the serveapi `context`
  field. Retrieval that a tool owns (e.g. exploitrank's CVE/actor linker) lives in
  the tool; `contextwindow.select` is the generic lexical fallback.
- **A provider** — any OpenAI-compatible endpoint is just a `providers:` profile
  in config. `mock` is the offline one. (A non-OpenAI backend is a bigger change —
  see §8.)
- **A CLI command** — add `cmd_x` + a subparser in `__main__.py` (lazy-import the
  feature module to keep startup light).

---

## 6. Integration surface (how the tools plug in)

```mermaid
flowchart LR
    subgraph Python hosts
        LS[learnscope ai.chat]
        JS[jobscope ai.chat]
        CB[claudebudget ai.chat]
    end
    GO[exploitrank Go\ninternal/ai.Complete]

    LS -->|import quorum.api.chat| API[api.py]
    JS -->|import quorum.api.chat| API
    CB -. ImportError → own path .-> API
    GO -->|HTTP /v1/chat/completions| SRV[serveapi.py]
    API --> ORCH[orchestrator]
    SRV --> ORCH
```

- **Python hosts** delegate at the top of their `ai.chat` to `quorum.api.chat`
  (guarded by `ImportError` + an `enabled` gate), forwarding `history`/`context`.
  If quorum is absent/disabled they fall back to their single-model path.
- **Non-Python hosts** (exploitrank) point an OpenAI-compatible client at
  `serve --api`; the request's `model` field selects the strategy, and an optional
  `context` field carries grounding docs.

---

## 7. Configuration model

One `DEFAULT_CONFIG` dict, deep-merged with the user's `config.yaml`. Secrets stay
in env vars named by each provider profile. Feature flags are **default-off**, so
a fresh config behaves like the pre-feature engine. Current sections: `council`,
`providers`, `run`, `promptsmith`, `judge`, `cost`, `context`, `output`.

---

## 8. Modularity assessment — evolving as features land

### Already modular (keep leaning on these)
- **Strategy registry + entry points** — the model to copy for other pluggable
  concerns.
- **`RunOptions` + the `hooks` pipeline** — new *knobs* land in one dataclass;
  new *stages* attach around the strategy without editing `run_session`.
- **Provider = the single network seam** — makes the whole engine offline-testable.
- **Stateless core + caller-owned state** — context/history are inputs, not
  engine memory.
- **Executable contract** — `selftest` + the mock provider force every feature to
  be offline-verifiable.

### Growing pains & recommended refactors
Ordered by payoff as more features arrive. Each is optional and independently
shippable; none changes behavior.

| # | Smell (today) | Move | Files | Priority |
|---|---|---|---|---|
| 1 | ~~`run.*` re-parsed in every strategy~~ | **Shipped**: typed `RunOptions` resolved once in the orchestrator, hung on `Context` | `strategies/__init__.py`, `orchestrator.py`, `strategies/*` | ✅ done |
| 2 | ~~orchestrator hard-codes its stages~~ | **Shipped**: pre/post `hooks` around the strategy | `hooks.py`, `orchestrator.py` | ✅ done |
| 3 | ~~lexical text-scoring split across callers~~ | **Shipped**: a `scoring/` package -- shared dependency-free primitives (`tokens`, `overlap_coeff`, `jaccard`) + a `Scorer` protocol & registry (built-in `lexical`, `quorum.scorers` entry points); `contextwindow.select` and `judge.consensus_reached` share them. The LLM rubric-judge / reference-grader can join the registry later | `scoring/`, `contextwindow.py`, `judge.py` | ✅ done |
| 4 | ~~`api.build_config` and serveapi request-mapping both translate *external → quorum*~~ | **Shipped**: a `quorum/adapters.py` entry-layer helper -- `host_config` (host `ai:`/`quorum:` -> quorum config), `split_messages` (OpenAI messages -> system/history/last_user), `select_strategy` (model field -> strategy). `api.build_config` is a thin wrapper and `serveapi._split`/`complete_chat` delegate, so both surfaces share one implementation | `adapters.py`, `api.py`, `serveapi.py` | ✅ done |
| 5 | ~~`prompts.py` is a flat grab-bag; strategy-specific builders (e.g. `challenge`) bloat it~~ | **Shipped**: a `prompts/` package split by concern -- the shared DATA/LLM01 framing helpers + generic builders (`propose`/`revise`/`self_refine`/`revise_from_draft`) in `base`, and the strategy-specific builders alongside their strategy (`debate.challenge`, `council.review`/`synthesize`, `moa.moa_layer`/`aggregate`). `__init__` re-exports every builder + SYSTEM constant, so `prompts.<name>` (and the `mock` sentinels) resolve byte-identically -- no call site changed | `prompts/`, `strategies/*` | ✅ done |
| 6 | `provider.py` bundles routing + transport (retry/fallback/json-mode) + accounting | Split a **transport** layer from a `Provider` protocol so alternate backends (embeddings, streaming, optional litellm) register like strategies | `provider.py` | Low* |
| 7 | ~~No config validation — a mistyped key silently no-ops~~ | **Shipped**: `config.validate_config` reports unknown key-paths and `load_config(warn=True)` prints them (open subtrees `providers.*`/`cost.pricing.*`/`judge.rubric.*` allowed); the CLI enables it. Never fatal | `config.py` | ✅ done |
| 8 | ~~`emit` is an ad-hoc string logger~~ | **Shipped**: a leaf `events.py` (`Event` + `render`/`coerce`); `run_session(on_event=...)` streams typed events (`phase`/`round`/`result`/`member_failed`/`done`) while plain-string emits are wrapped as `log` events, so the CLI output is byte-identical and no call site broke. `ctx.event(kind, msg, **data)` is the strategy-side helper | `events.py`, `orchestrator.py`, `strategies/*` | ✅ done |

\* Low unless a non-OpenAI or multi-backend provider lands on the roadmap — then #6 jumps to High.

### Suggested near-term order
**#1 (RunOptions)**, **#2 (hook pipeline)**, **#3 (scoring package)**,
**#4 (adapters)**, and **#5 (prompts package)** are shipped -- together they cover
the commonest shapes of a new feature (a new *knob* -> #1, a new *stage* -> #2, a
new *text-scoring measure* -> #3, a new *host integration* -> #4, a new *prompt
builder* -> #5). Next up: **#6 (provider transport split)** if/when a non-OpenAI
or multi-backend provider lands (it jumps to High then).

---

## 9. The "add a feature" checklist

1. **Default-off** config in `DEFAULT_CONFIG` + `config.example.yaml`.
2. **Mock support** — the `mock` provider must exercise it offline.
3. **A `selftest` check** and a **pytest** (offline, deterministic).
4. **No new hard dependency** (optional extras only, guarded).
5. **Respect the layers** — reasoning services don't import strategies; domain
   imports nothing above it.
6. If it touches the embed path, keep `api.chat` **signature-compatible** so the
   sibling tools' delegation keeps working.
