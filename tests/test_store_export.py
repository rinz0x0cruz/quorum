import os

from quorum import exporter, orchestrator, render
from quorum.store import Store
from tests.helpers import mock_cfg


def test_session_round_trip_and_exports(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        sess = orchestrator.run_session(cfg, "explain something", store=store, strategy="refine")
        d = store.get_session(sess.id)
        assert d and d["final"]

        md = tmp_path / "out.md"
        assert exporter.run(cfg, store, fmt="md", session_id=sess.id, out=str(md)) == 0
        assert "Final answer" in md.read_text(encoding="utf-8")

        csv_path = tmp_path / "out.csv"
        assert exporter.run(cfg, store, fmt="csv", session_id=sess.id, out=str(csv_path)) == 0
        assert csv_path.exists()

        js = tmp_path / "out.json"
        assert exporter.run(cfg, store, fmt="json", session_id=sess.id, out=str(js)) == 0
        assert js.exists()


def test_dashboard_is_self_contained(tmp_path):
    cfg = mock_cfg(str(tmp_path / "t.db"))
    with Store(cfg["output"]["db_path"]) as store:
        orchestrator.run_session(cfg, "x", store=store, strategy="debate")
        path = render.build(cfg, store)
    assert os.path.exists(path)
    html = open(path, "r", encoding="utf-8").read()
    assert "const D =" in html and "__DATA__" not in html
