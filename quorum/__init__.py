"""quorum -- multi-model deliberation for prompt + solution refinement.

Several AI models are put "in conversation": a prompt is first designed and
refined (phase 1), then a solution is bounced between the models -- proposing,
critiquing, and revising -- while a judge scores each round until it is "good
enough" (phase 2). Three deliberation strategies ship in the box (debate+judge,
council+chairman, mixture-of-agents) plus cheap baselines, and a benchmark
harness compares them on your own tasks.

The engine (rounds, judging, stopping, storage, rendering, cost accounting, the
benchmark) is deterministic and runs fully offline via the built-in ``mock``
provider. Live deliberation talks to any OpenAI-compatible endpoint (OpenRouter,
a local Ollama, Groq, OpenAI, ...); keys come from the environment and nothing
is uploaded anywhere but the model endpoints you configure.
"""
from __future__ import annotations

__version__ = "0.1.0"
