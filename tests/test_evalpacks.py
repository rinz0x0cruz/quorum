from __future__ import annotations

import copy
from pathlib import Path

import yaml

from quorum.evalpacks import load_split, split_fingerprint, verify_pack


SHIPPED_PACKS = {
    "coding-humaneval-static",
    "extraction-banking77",
    "reasoning-math-gsm8k",
    "research-openalex",
    "security-cisa-kev",
    "writing-authored",
}


def _write_pack(root: Path) -> tuple[dict, dict[str, list[dict]]]:
    splits = {
        "train": [{"id": "train-1", "task": "Classify this message.", "answer": "billing", "match": "exact"}],
        "validation": [{"id": "validation-1", "task": "What is 2 + 2?", "answer": "4", "match": "numeric"}],
        "promotion_test": [{"id": "test-1", "task": "Is 8 even?", "answer": "yes", "match": "boolean"}],
    }
    manifest = {
        "schema_version": 1,
        "id": "fixture-pack",
        "version": "1.0.0",
        "use_case": "extraction_classification",
        "license": {"spdx": "CC-BY-4.0", "redistribution": True, "training": True},
        "sources": [{"id": "fixture", "url": "https://example.invalid/data", "revision": "fixture-v1"}],
        "splits": {},
    }
    root.mkdir()
    for split_name, tasks in splits.items():
        path = root / f"{split_name}.yaml"
        path.write_text(yaml.safe_dump({"tasks": tasks}, sort_keys=False), encoding="utf-8")
        manifest["splits"][split_name] = {
            "path": path.name,
            "fingerprint": split_fingerprint(tasks),
        }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest, splits


def test_verify_and_load_pack(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    _, splits = _write_pack(root)

    result = verify_pack(root)

    assert result.ok, result.errors
    assert result.pack_id == "fixture-pack"
    assert load_split(root, "validation") == splits["validation"]


def test_verify_rejects_stale_fingerprint_and_split_leakage(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    manifest, splits = _write_pack(root)
    leaked = copy.deepcopy(splits["train"][0])
    leaked["id"] = "test-leak"
    (root / "promotion_test.yaml").write_text(
        yaml.safe_dump({"tasks": [leaked]}, sort_keys=False), encoding="utf-8"
    )
    manifest["splits"]["validation"]["fingerprint"] = "sha256:stale"
    manifest["splits"]["promotion_test"]["fingerprint"] = split_fingerprint([leaked])
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    result = verify_pack(root)

    assert not result.ok
    assert any("fingerprint mismatch" in error for error in result.errors)
    assert any("duplicates" in error and "train" in error and "promotion_test" in error for error in result.errors)


def test_shipped_packs_verify_and_keep_sealed_splits_out_of_training() -> None:
    root = Path(__file__).resolve().parents[1] / "evals" / "packs"
    manifests = sorted(root.glob("*/manifest.yaml"))

    assert {manifest.parent.name for manifest in manifests} == SHIPPED_PACKS
    for manifest_path in manifests:
        result = verify_pack(manifest_path)
        assert result.ok, (manifest_path, result.errors)
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        governance = manifest["governance"]
        assert governance["training_splits"] == ["train"]
        assert "promotion_test" in governance["sealed_splits"]
        for split in ("train", "validation", "promotion_test"):
            tasks = load_split(manifest_path, split)
            assert len(tasks) >= 2
            assert all(task.get("source_id") for task in tasks)


def test_humaneval_is_evaluation_only_and_never_executed() -> None:
    root = Path(__file__).resolve().parents[1] / "evals" / "packs" / "coding-humaneval-static"
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    train = load_split(root, "train")
    validation = load_split(root, "validation")

    assert manifest["governance"]["evaluation_only_sources"] == ["humaneval"]
    assert manifest["governance"]["executes_untrusted_code"] is False
    assert all(task["source_id"].startswith("quorum-authored-") for task in train)
    assert all(task["source_id"].startswith("HumanEval/") for task in validation)
    assert all("no_execution" in task["tags"] for task in validation)