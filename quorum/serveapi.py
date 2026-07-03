"""OpenAI-compatible local API: ``POST /v1/chat/completions`` -> a deliberation.

Lets ANY OpenAI-compatible client -- including non-Python tools like the Go
``exploitrank`` -- use quorum as a drop-in "model" by pointing its ``base_url``
at this server. The client's ``system`` + ``user`` messages drive the
deliberation; the final answer is returned in the standard OpenAI response shape,
so existing clients (and their caching/gating) work unchanged.

Binds to 127.0.0.1 only. An optional bearer token can be required. The request's
``model`` field selects the strategy when it names one (debate/council/moa/refine/
ensemble), otherwise the server's configured ``run.strategy`` is used.
"""
from __future__ import annotations

import copy
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import orchestrator
from .strategies import available as strategies_available


def _extract(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return system, (users[-1] if users else "")


def complete_chat(cfg: dict, req: dict) -> tuple[int, dict[str, Any]]:
    """Pure request->response (no HTTP), so it is unit-testable offline."""
    messages = req.get("messages") or []
    system, user = _extract(messages)
    if not user:
        return 400, {"error": {"message": "no user message"}}

    strategies = set(strategies_available())
    default_strategy = (cfg.get("run", {}) or {}).get("strategy", "refine")
    model = req.get("model", "") or ""
    strategy = model if model in strategies else default_strategy

    rcfg = copy.deepcopy(cfg)  # per-request: run_session mutates run.strategy
    try:
        sess = orchestrator.run_session(rcfg, user, solve_prompt=system,
                                        promptsmith_on=False, strategy=strategy, verbose=False)
    except Exception as e:  # noqa: BLE001 - surface as a gateway error, never crash the server
        return 502, {"error": {"message": f"deliberation failed: {e}"}}

    final = sess.final or ""
    return 200, {
        "id": "quorum-" + sess.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"quorum/{sess.strategy}",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": final},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": sess.tokens_in, "completion_tokens": sess.tokens_out,
                  "total_tokens": sess.tokens_in + sess.tokens_out},
    }


def run(cfg: dict, port: int = 8802, token: str = "") -> int:
    default_strategy = (cfg.get("run", {}) or {}).get("strategy", "refine")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth_ok(self) -> bool:
            return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

        def do_GET(self) -> None:
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok"})
            if path == "/v1/models":
                data = [{"id": s, "object": "model"} for s in sorted(strategies_available())]
                return self._send(200, {"object": "list", "data": data})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            if not self._auth_ok():
                return self._send(401, {"error": {"message": "unauthorized"}})
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            except (ValueError, TypeError):
                return self._send(400, {"error": {"message": "invalid JSON"}})
            code, obj = complete_chat(cfg, req)
            self._send(code, obj)

        def log_message(self, *args) -> None:  # keep the console quiet
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        note = ", token required" if token else ""
        print(f"  quorum OpenAI-compatible API at http://127.0.0.1:{port}/v1  "
              f"(strategy={default_strategy}{note})  Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped")
    return 0
