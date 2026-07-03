from quorum import hooks, orchestrator
from quorum.store import Store
from tests.helpers import mock_cfg


def test_pre_and_post_hooks_fire(tmp_path):
    hooks.clear()
    fired = {"pre": 0, "post": 0}

    @hooks.register_pre
    def _pre(ctx):
        fired["pre"] += 1
        assert ctx.opts is not None and ctx.task  # pre-hook sees a built Context

    @hooks.register_post
    def _post(ctx):
        fired["post"] += 1
        assert ctx.session.final is not None  # post-hook sees the finished session

    try:
        cfg = mock_cfg(str(tmp_path / "t.db"))
        with Store(cfg["output"]["db_path"]) as store:
            orchestrator.run_session(cfg, "hook me", store=store, strategy="refine",
                                     promptsmith_on=False)
        assert fired == {"pre": 1, "post": 1}
    finally:
        hooks.clear()


def test_pre_hook_can_mutate_prompt(tmp_path):
    hooks.clear()

    @hooks.register_pre
    def _pre(ctx):
        ctx.prompt = ctx.prompt + "\n\n[injected by hook]"

    try:
        cfg = mock_cfg(str(tmp_path / "t.db"))
        with Store(cfg["output"]["db_path"]) as store:
            sess = orchestrator.run_session(cfg, "x", store=store, strategy="refine",
                                            promptsmith_on=False)
        assert sess.final  # still produces an answer with the mutated prompt
    finally:
        hooks.clear()


def test_no_hooks_by_default(tmp_path):
    hooks.clear()
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "x", store=store, strategy="refine",
                                        promptsmith_on=False)
    assert sess.final and sess.status == "ok"
