from __future__ import annotations

import hashlib
from pathlib import Path

from quorum import datasets
from quorum.__main__ import build_parser


def test_source_registry_pins_immutable_files() -> None:
    immutable = [source for source in datasets.SOURCE_FILES if not source.mutable]
    assert immutable
    assert all(source.sha256 and len(source.sha256) == 64 for source in immutable)
    assert all(source.size_bytes and source.size_bytes > 0 for source in immutable)
    assert all(len(source.revision) == 40 for source in immutable)
    mutable = [source for source in datasets.SOURCE_FILES if source.mutable]
    assert mutable and all(source.sha256 is None and source.size_bytes is None for source in mutable)
    assert datasets.source_ids() == ["banking77", "cisa-kev", "gsm8k", "humaneval", "openalex"]


def test_fetch_sources_accepts_verified_cache_without_network(tmp_path: Path) -> None:
    payload = b'{"value": 42}\n'
    source = datasets.SourceFile(
        source_id="fixture",
        path="fixture/data.json",
        url="https://example.invalid/data.json",
        revision="fixture-v1",
        license_spdx="CC0-1.0",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    target = tmp_path / source.path
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    records = datasets.fetch_sources(tmp_path, ["fixture"], files=[source])

    assert records[0]["status"] == "cached"
    assert records[0]["observed_sha256"] == source.sha256
    assert (tmp_path / "sources.lock.json").exists()


def test_verify_sources_never_downloads_missing_files(tmp_path: Path, monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fail_network)

    try:
        datasets.verify_sources(tmp_path, ["gsm8k"])
    except datasets.DatasetError as exc:
        assert "missing source" in str(exc)
    else:
        raise AssertionError("missing sources must fail verification")


def test_packs_cli_namespace_is_registered() -> None:
    parser = build_parser()

    fetch = parser.parse_args(["packs", "fetch", "--source", "gsm8k"])
    prepare = parser.parse_args(["packs", "prepare"])
    verify = parser.parse_args(["packs", "verify", "evals/packs/example"])

    assert fetch.func.__name__ == "cmd_packs_fetch"
    assert prepare.func.__name__ == "cmd_packs_prepare"
    assert verify.func.__name__ == "cmd_packs_verify"


def test_prepare_smoke_fixtures_is_reproducible(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1] / "data" / "packs" / "sources"
    try:
        datasets.verify_sources(source_root)
    except datasets.DatasetError:
        return
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = datasets.prepare_smoke_fixtures(source_root, first_root)
    second = datasets.prepare_smoke_fixtures(source_root, second_root)

    assert [path.name for path in first] == [path.name for path in second]
    for first_path, second_path in zip(first, second):
        assert (first_path / "manifest.yaml").read_bytes() == (second_path / "manifest.yaml").read_bytes()


def test_banking77_dedup_preserves_official_test_precedence(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    banking_root = source_root / "banking77"
    banking_root.mkdir(parents=True)
    (banking_root / "train.csv").write_text(
        "text,category\nSame request,intent_a\n same   request ,intent_a\nTrain only,intent_b\n",
        encoding="utf-8",
    )
    (banking_root / "test.csv").write_text(
        "text,category\nSAME REQUEST,intent_a\nTest only,intent_c\nTest only,intent_c\n",
        encoding="utf-8",
    )

    train, test = datasets._banking77_rows(source_root)

    assert [row["text"] for row in train] == ["Train only"]
    assert [row["text"] for row in test] == ["SAME REQUEST", "Test only"]