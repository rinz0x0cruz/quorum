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

import concurrent.futures
import copy
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import adapters, events, orchestrator
# re-exported for the API/test-suite (serveapi._split); see tests/test_serveapi.py
from .adapters import split_messages as _split  # noqa: F401
from .strategies import available as strategies_available

MAX_BODY = 1_000_000  # 1 MB request cap
_DONE = object()      # queue sentinel: the deliberation worker has finished


def complete_chat(cfg: dict, req: dict) -> tuple[int, dict[str, Any]]:
    """Pure request->response (no HTTP), so it is unit-testable offline."""
    messages = req.get("messages") or []
    system, history, user = adapters.split_messages(messages)
    if not user:
        return 400, {"error": {"message": "no user message"}}
    # Optional, non-standard grounding docs (OpenAI clients simply omit this).
    context = req.get("context") or None

    strategy = adapters.select_strategy(req.get("model", ""), cfg)

    rcfg = copy.deepcopy(cfg)  # per-request: run_session mutates run.strategy
    try:
        sess = orchestrator.run_session(rcfg, user, solve_prompt=system,
                                        history=history or None, context=context,
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


def _stream_events(cfg: dict, req: dict, write, *, request_timeout: float = 120.0) -> None:
    """Drive an SSE stream via ``write(text)``: **live** progress as SSE comment
    lines while the deliberation runs (round scores, phase, stop), then the final
    answer as an OpenAI ``chat.completion.chunk`` delta + ``[DONE]``.

    Progress is sent as SSE comments (``: ...``) which OpenAI stream clients ignore
    but raw SSE consumers can read, so the response stays OpenAI-compatible.
    ``write`` is any sink (the HTTP socket, or a list in tests) -> HTTP-free + testable.
    """
    messages = req.get("messages") or []
    system, history, user = adapters.split_messages(messages)
    context = req.get("context") or None
    strategy = adapters.select_strategy(req.get("model", ""), cfg)
    rcfg = copy.deepcopy(cfg)
    created = int(time.time())

    q: "queue.Queue" = queue.Queue()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["sess"] = orchestrator.run_session(
                rcfg, user, solve_prompt=system, history=history or None, context=context,
                promptsmith_on=False, strategy=strategy, verbose=False,
                on_event=q.put)
        except Exception as e:  # noqa: BLE001 - surface as an error chunk, never crash
            result["error"] = str(e)
        finally:
            q.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()

    deadline = time.monotonic() + request_timeout
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            item = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if item is _DONE:
            break
        write(f": {events.render(item)}\n\n")

    sess = result.get("sess")
    model = f"quorum/{sess.strategy}" if sess else f"quorum/{strategy}"
    sid = "quorum-" + (sess.id if sess else "stream")
    if timed_out:
        content, err = "", "deliberation timed out"
    elif "error" in result:
        content, err = "", "deliberation failed: " + result["error"]
    else:
        content, err = (sess.final or ""), None
    if err:
        write(f": error: {err}\n\n")
    base = {"id": sid, "object": "chat.completion.chunk", "created": created, "model": model}
    write("data: " + json.dumps({**base, "choices": [
        {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}) + "\n\n")
    write("data: " + json.dumps({**base, "choices": [
        {"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n")
    write("data: [DONE]\n\n")


def make_server(cfg: dict, host: str = "127.0.0.1", port: int = 8802, token: str = "",
                request_timeout: float = 120.0) -> ThreadingHTTPServer:

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_stream_headers(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

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
                if n > MAX_BODY:
                    return self._send(413, {"error": {"message": "payload too large"}})
                req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            except (ValueError, TypeError):
                return self._send(400, {"error": {"message": "invalid JSON"}})
            if req.get("stream"):
                # validate up front so a bad request still gets a clean 400, not a 200 stream
                if not adapters.split_messages(req.get("messages") or [])[2]:
                    return self._send(400, {"error": {"message": "no user message"}})
                self._send_stream_headers()

                def _write(s: str) -> None:
                    try:
                        self.wfile.write(s.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionError, OSError):
                        pass
                _stream_events(cfg, req, _write, request_timeout=request_timeout)
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(complete_chat, cfg, req)
                try:
                    code, obj = fut.result(timeout=request_timeout)
                except concurrent.futures.TimeoutError:
                    return self._send(504, {"error": {"message": "deliberation timed out"}})
            self._send(code, obj)

        def log_message(self, *args) -> None:  # keep the console quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def run(cfg: dict, port: int = 8802, token: str = "", host: str = "127.0.0.1",
        request_timeout: float = 120.0) -> int:
    default_strategy = (cfg.get("run", {}) or {}).get("strategy", "refine")
    insecure = host not in ("127.0.0.1", "localhost", "::1") and not token
    httpd = make_server(cfg, host=host, port=port, token=token, request_timeout=request_timeout)
    note = ", token required" if token else ""
    warn = "\n  [!] bound to a non-loopback host WITHOUT a token -- set --token" if insecure else ""
    print(f"  quorum OpenAI-compatible API at http://{host}:{port}/v1  "
          f"(strategy={default_strategy}{note}){warn}  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
    return 0
