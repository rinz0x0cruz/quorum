import json
import threading
import urllib.error
import urllib.request

from quorum import serveapi
from tests.helpers import mock_cfg


def _serve(cfg, token=""):
    httpd = serveapi.make_server(cfg, host="127.0.0.1", port=0, token=token)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _post(port, body, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode()


def test_http_round_trip(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    httpd, port = _serve(cfg)
    try:
        code, text = _post(port, {"model": "refine", "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"}]})
        obj = json.loads(text)
        assert code == 200
        assert obj["object"] == "chat.completion"
        assert obj["model"] == "quorum/refine"
        assert obj["choices"][0]["message"]["content"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_stream_sse(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    httpd, port = _serve(cfg)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "refine", "stream": True,
                             "messages": [{"role": "user", "content": "hi"}]}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            ctype = r.headers.get("Content-Type", "")
            payload = r.read().decode()
        assert "text/event-stream" in ctype
        assert "data: " in payload and "[DONE]" in payload
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_auth_401(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    httpd, port = _serve(cfg, token="secret")
    try:
        try:
            _post(port, {"messages": [{"role": "user", "content": "x"}]})
            raise AssertionError("expected 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()
