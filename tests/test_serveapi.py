from quorum import serveapi
from tests.helpers import mock_cfg


def test_complete_chat_returns_openai_shape(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    req = {"model": "refine", "messages": [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Say hello."}]}
    code, obj = serveapi.complete_chat(cfg, req)
    assert code == 200
    assert obj["object"] == "chat.completion"
    assert obj["choices"][0]["message"]["role"] == "assistant"
    assert obj["choices"][0]["message"]["content"]
    assert obj["model"] == "quorum/refine"
    assert obj["usage"]["total_tokens"] > 0


def test_model_field_selects_strategy(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    req = {"model": "debate", "messages": [{"role": "user", "content": "x"}]}
    code, obj = serveapi.complete_chat(cfg, req)
    assert code == 200 and obj["model"] == "quorum/debate"


def test_missing_user_message_is_400(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    code, obj = serveapi.complete_chat(cfg, {"messages": [{"role": "system", "content": "s"}]})
    assert code == 400 and "error" in obj


def test_extract_takes_last_user_and_all_system():
    system, user = serveapi._extract([
        {"role": "system", "content": "a"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "second"},
    ])
    assert system == "a" and user == "second"
