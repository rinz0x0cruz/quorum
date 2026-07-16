"""Reproducible dataset downloads and evaluation-pack preparation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .evalpacks import split_fingerprint, verify_pack


@dataclass(frozen=True)
class SourceFile:
    """One pinned or explicitly mutable upstream dataset file."""

    source_id: str
    path: str
    url: str
    revision: str
    license_spdx: str
    sha256: str | None = None
    size_bytes: int | None = None
    mutable: bool = False


_GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
_HUMANEVAL_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
_BANKING77_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
_CISA_KEV_REVISION = "46887d9ebae72243eb0e476876a75152e63e90d9"

SOURCE_FILES = (
    SourceFile(
        "gsm8k", "gsm8k/train.jsonl",
        f"https://raw.githubusercontent.com/openai/grade-school-math/{_GSM8K_REVISION}/"
        "grade_school_math/data/train.jsonl",
        _GSM8K_REVISION, "MIT",
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465", 4_166_206,
    ),
    SourceFile(
        "gsm8k", "gsm8k/test.jsonl",
        f"https://raw.githubusercontent.com/openai/grade-school-math/{_GSM8K_REVISION}/"
        "grade_school_math/data/test.jsonl",
        _GSM8K_REVISION, "MIT",
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14", 749_738,
    ),
    SourceFile(
        "humaneval", "humaneval/HumanEval.jsonl.gz",
        f"https://raw.githubusercontent.com/openai/human-eval/{_HUMANEVAL_REVISION}/"
        "data/HumanEval.jsonl.gz",
        _HUMANEVAL_REVISION, "MIT",
        "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef", 44_877,
    ),
    SourceFile(
        "banking77", "banking77/train.csv",
        f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{_BANKING77_REVISION}/"
        "banking_data/train.csv",
        _BANKING77_REVISION, "CC-BY-4.0",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b", 839_073,
    ),
    SourceFile(
        "banking77", "banking77/test.csv",
        f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{_BANKING77_REVISION}/"
        "banking_data/test.csv",
        _BANKING77_REVISION, "CC-BY-4.0",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d", 239_961,
    ),
    SourceFile(
        "banking77", "banking77/categories.json",
        f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{_BANKING77_REVISION}/"
        "banking_data/categories.json",
        _BANKING77_REVISION, "CC-BY-4.0",
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63", 2_036,
    ),
    SourceFile(
        "cisa-kev", "cisa-kev/known_exploited_vulnerabilities.json",
        f"https://raw.githubusercontent.com/cisagov/kev-data/{_CISA_KEV_REVISION}/"
        "known_exploited_vulnerabilities.json",
        _CISA_KEV_REVISION, "CC0-1.0",
        "769c7c2dbf9f55343e50298691e47932140f25fda3e820e4b1c57ba36fcc6c6b", 1_544_013,
    ),
    SourceFile(
        "cisa-kev", "cisa-kev/known_exploited_vulnerabilities_schema.json",
        f"https://raw.githubusercontent.com/cisagov/kev-data/{_CISA_KEV_REVISION}/"
        "known_exploited_vulnerabilities_schema.json",
        _CISA_KEV_REVISION, "CC0-1.0",
        "577f4ccc06b7b7c6a109e1a0d6457a26db7fc5219398ff2e287b9a7e14e2d9ef", 3_407,
    ),
    SourceFile(
        "openalex", "openalex/W4400141368.json",
        "https://api.openalex.org/works/W4400141368",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
    SourceFile(
        "openalex", "openalex/W4376122773.json",
        "https://api.openalex.org/works/W4376122773",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
    SourceFile(
        "openalex", "openalex/W4392971790.json",
        "https://api.openalex.org/works/W4392971790",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
    SourceFile(
        "openalex", "openalex/W4396570275.json",
        "https://api.openalex.org/works/W4396570275",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
    SourceFile(
        "openalex", "openalex/W4399511904.json",
        "https://api.openalex.org/works/W4399511904",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
    SourceFile(
        "openalex", "openalex/W4389217455.json",
        "https://api.openalex.org/works/W4389217455",
        "observed-2026-07-15", "CC0-1.0", mutable=True,
    ),
)

_RESEARCH_WORK_SPLITS = {
    "train": (
        "W4400141368",
        "W4376122773",
        "routing decisions should be evidence-driven",
    ),
    "validation": (
        "W4392971790",
        "W4396570275",
        "evaluation should measure both routing and judge diversity",
    ),
    "promotion_test": (
        "W4399511904",
        "W4389217455",
        "multi-model gains depend on the aggregation protocol",
    ),
}

_AUTHORED_CODING_TASKS = [
    {
        "id": "quorum-coding-clamp",
        "task": (
            "Choose the one-line replacement that correctly clamps value between lower and upper. "
            "Do not execute any candidate. End with \"Answer: <letter>\".\n\n"
            "def clamp(value, lower, upper):\n"
            "    return min(lower, max(value, upper))\n\n"
            "A) return min(upper, max(lower, value))\n"
            "B) return max(upper, min(lower, value))\n"
            "C) return min(lower, min(upper, value))\n"
            "D) return max(lower, max(upper, value))"
        ),
        "reference": "A",
        "match": "choice",
        "source_id": "quorum-authored-coding-001",
        "tags": ["coding", "static", "bug_fix", "no_execution"],
    },
    {
        "id": "quorum-coding-output",
        "task": (
            "Reason about this Python snippet without executing it. Which output is produced? "
            "End with \"Answer: <letter>\".\n\n"
            "def unique(items):\n"
            "    return list(dict.fromkeys(items))\n\n"
            "print(unique(['b', 'a', 'b']))\n\n"
            "A) ['a', 'b']\nB) ['b', 'a']\nC) ['b', 'a', 'b']\nD) {'a', 'b'}"
        ),
        "reference": "B",
        "match": "choice",
        "source_id": "quorum-authored-coding-002",
        "tags": ["coding", "static", "expected_output", "no_execution"],
    },
]

_AUTHORED_WRITING_SPLITS = {
    "train": [
        {
            "id": "writing-status-summary",
            "task": (
                "Summarize this project update in one neutral sentence and end with the exact phrase "
                "\"launch remains on schedule\". DATA: The migration completed Tuesday. Two low-risk "
                "documentation items remain. No launch blockers are open."
            ),
            "reference": "launch remains on schedule",
            "match": "contains",
            "source_id": "quorum-authored-writing-001",
            "tags": ["writing", "summarization", "status"],
        },
        {
            "id": "writing-refund-rewrite",
            "task": (
                "Rewrite this support note with a calm, direct tone in at most two sentences and include "
                "the exact phrase \"refund will arrive within five business days\". DATA: We approved "
                "the request today. Bank processing may take up to five business days."
            ),
            "reference": "refund will arrive within five business days",
            "match": "contains",
            "source_id": "quorum-authored-writing-002",
            "tags": ["writing", "rewrite", "customer_support"],
        },
    ],
    "validation": [
        {
            "id": "writing-incident-summary",
            "task": (
                "Write a concise executive incident summary using only the data below and include the "
                "exact phrase \"service restored in 23 minutes\". DATA: Alerts began at 09:14 UTC. "
                "Traffic was shifted at 09:29. Recovery completed at 09:37. No data was lost."
            ),
            "reference": "service restored in 23 minutes",
            "match": "contains",
            "source_id": "quorum-authored-writing-003",
            "tags": ["writing", "summarization", "incident"],
        },
        {
            "id": "writing-release-note",
            "task": (
                "Turn the data into one user-facing release-note sentence and include the exact phrase "
                "\"CSV exports now preserve leading zeros\". DATA: The export formatter previously "
                "coerced identifier columns to numbers. Identifier columns are now emitted as text."
            ),
            "reference": "CSV exports now preserve leading zeros",
            "match": "contains",
            "source_id": "quorum-authored-writing-004",
            "tags": ["writing", "release_note"],
        },
    ],
    "promotion_test": [
        {
            "id": "writing-policy-summary",
            "task": (
                "Summarize the policy in one sentence without adding exceptions and include the exact "
                "phrase \"manual approval is required\". DATA: Automated checks may recommend a model. "
                "Only a reviewed holdout result can activate it in a production profile."
            ),
            "reference": "manual approval is required",
            "match": "contains",
            "source_id": "quorum-authored-writing-005",
            "tags": ["writing", "summarization", "policy"],
        },
        {
            "id": "writing-risk-brief",
            "task": (
                "Write a neutral two-sentence risk brief and include the exact phrase \"the evidence is "
                "inconclusive\". DATA: One small test improved latency by 8 percent. Error rate was "
                "unchanged, but the sample was too small for a confidence interval."
            ),
            "reference": "the evidence is inconclusive",
            "match": "contains",
            "source_id": "quorum-authored-writing-006",
            "tags": ["writing", "risk", "neutrality"],
        },
    ],
}


class DatasetError(ValueError):
    """Raised when a source or prepared dataset violates its manifest."""


def source_ids() -> list[str]:
    """Return the registered upstream dataset IDs."""
    return sorted({source.source_id for source in SOURCE_FILES})


def fetch_sources(
    root: str | Path = "data/packs/sources",
    selected: Iterable[str] | None = None,
    *,
    force: bool = False,
    files: Iterable[SourceFile] | None = None,
) -> list[dict[str, Any]]:
    """Fetch selected sources, enforcing pinned size and SHA-256 metadata."""
    root_path = Path(root)
    requested = set(selected or source_ids())
    known = source_ids() if files is None else sorted({item.source_id for item in files})
    unknown = requested.difference(known)
    if unknown:
        raise DatasetError(f"unknown dataset source(s): {', '.join(sorted(unknown))}")

    lock_records = _lock_records(root_path / "sources.lock.json")
    records: list[dict[str, Any]] = []
    for source in files or SOURCE_FILES:
        if source.source_id not in requested:
            continue
        target = root_path / source.path
        target.parent.mkdir(parents=True, exist_ok=True)
        status = "cached"
        if force or not target.exists():
            _download(source, target)
            status = "downloaded"
        observed_sha = _sha256(target)
        observed_size = target.stat().st_size
        _check_observed(source, observed_sha, observed_size)
        if source.mutable and not force and source.path in lock_records:
            _check_locked(source.path, lock_records[source.path], observed_sha, observed_size)
        _validate_source(source, target)
        records.append({
            **asdict(source),
            "observed_sha256": observed_sha,
            "observed_size_bytes": observed_size,
            "status": status,
        })

    root_path.mkdir(parents=True, exist_ok=True)
    for record in records:
        lock_records[record["path"]] = record
    lock = {
        "schema_version": 1,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(lock_records.values(), key=lambda item: item["path"]),
    }
    _atomic_write(
        root_path / "sources.lock.json",
        json.dumps(lock, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return records


def verify_sources(
    root: str | Path = "data/packs/sources",
    selected: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify already-downloaded sources without network access."""
    root_path = Path(root)
    requested = set(selected or source_ids())
    unknown = requested.difference(source_ids())
    if unknown:
        raise DatasetError(f"unknown dataset source(s): {', '.join(sorted(unknown))}")
    lock_records = _lock_records(root_path / "sources.lock.json")
    records: list[dict[str, Any]] = []
    for source in SOURCE_FILES:
        if source.source_id not in requested:
            continue
        target = root_path / source.path
        _require(target)
        observed_sha = _sha256(target)
        observed_size = target.stat().st_size
        if source.mutable:
            locked = lock_records.get(source.path)
            if locked is None:
                raise DatasetError(
                    f"{source.path}: missing sources.lock.json entry; run `quorum packs fetch`"
                )
            _check_locked(source.path, locked, observed_sha, observed_size)
        else:
            _check_observed(source, observed_sha, observed_size)
        _validate_source(source, target)
        records.append({
            **asdict(source),
            "observed_sha256": observed_sha,
            "observed_size_bytes": observed_size,
            "status": "verified",
        })
    return records


def prepare_all(
    source_root: str | Path = "data/packs/sources",
    output_root: str | Path = "data/packs/prepared",
) -> list[Path]:
    """Prepare full local packs for all six supported use cases."""
    source_path = Path(source_root)
    output_path = Path(output_root)
    verify_sources(source_path)
    return [
        _prepare_gsm8k(source_path, output_path),
        _prepare_banking77(source_path, output_path),
        _prepare_humaneval(source_path, output_path),
        _prepare_cisa_kev(source_path, output_path),
        _prepare_openalex_research(source_path, output_path),
        _prepare_writing(output_path),
    ]


def prepare_smoke_fixtures(
    source_root: str | Path = "data/packs/sources",
    output_root: str | Path = "evals/packs",
    *,
    tasks_per_split: int = 2,
) -> list[Path]:
    """Build small, attributed fixture packs with the full-pack split policy."""
    if tasks_per_split < 1:
        raise DatasetError("tasks_per_split must be at least 1")
    source_path = Path(source_root)
    output_path = Path(output_root)
    verify_sources(source_path)

    gsm_train = _read_jsonl(source_path / "gsm8k/train.jsonl")
    gsm_test = _read_jsonl(source_path / "gsm8k/test.jsonl")
    gsm_splits = _partition_smoke(
        gsm_train, lambda row: "validation" if _bucket(row["question"], 10) == 0 else "train",
        _gsm8k_task, tasks_per_split,
        required=("train", "validation"),
    )
    gsm_splits["promotion_test"] = [_gsm8k_task(row) for row in gsm_test[:tasks_per_split]]

    bank_train, bank_test = _banking77_rows(source_path)
    bank_splits = _partition_smoke(
        bank_train, lambda row: "validation" if _bucket(row["text"], 10) == 0 else "train",
        _banking77_task, tasks_per_split,
        required=("train", "validation"),
    )
    bank_splits["promotion_test"] = [
        _banking77_task(row) for row in bank_test[:tasks_per_split]
    ]

    with (source_path / "cisa-kev/known_exploited_vulnerabilities.json").open(
        "r", encoding="utf-8"
    ) as handle:
        kev_rows = json.load(handle)["vulnerabilities"]

    def kev_split(row: dict[str, Any]) -> str:
        bucket = _bucket(row["cveID"], 10)
        return "promotion_test" if bucket == 0 else ("validation" if bucket == 1 else "train")

    kev_splits = _partition_smoke(kev_rows, kev_split, _cisa_kev_task, tasks_per_split)

    with gzip.open(source_path / "humaneval/HumanEval.jsonl.gz", "rt", encoding="utf-8") as handle:
        humaneval_rows = [json.loads(line) for line in handle if line.strip()]
    humaneval_splits = _partition_smoke(
        humaneval_rows,
        lambda row: "promotion_test" if _bucket(row["task_id"], 2) else "validation",
        _humaneval_static_task,
        tasks_per_split,
        required=("validation", "promotion_test"),
    )
    humaneval_splits["train"] = [dict(task) for task in _AUTHORED_CODING_TASKS]

    governance = {"training_splits": ["train"], "sealed_splits": ["promotion_test"]}
    return [
        _write_pack(
            output_path, "reasoning-math-gsm8k", "reasoning_math", "MIT", gsm_splits,
            [_source_manifest("gsm8k")], governance,
        ),
        _write_pack(
            output_path, "extraction-banking77", "extraction_classification", "CC-BY-4.0",
            bank_splits, [_source_manifest("banking77")], governance,
        ),
        _write_pack(
            output_path, "security-cisa-kev", "security_analysis", "CC0-1.0", kev_splits,
            [_source_manifest("cisa-kev")], governance,
        ),
        _write_pack(
            output_path, "coding-humaneval-static", "coding", "MIT", humaneval_splits,
            [_source_manifest("humaneval"), _authored_source("coding")],
            {**governance, "evaluation_only_sources": ["humaneval"],
             "executes_untrusted_code": False},
        ),
        _prepare_openalex_research(source_path, output_path),
        _prepare_writing(output_path),
    ]


def _download(source: SourceFile, target: Path) -> None:
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "quorum-dataset-fetch/0.1 (+https://github.com/rinz0x0cruz/quorum)"},
    )
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temp_path = Path(handle.name)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        observed_sha = _sha256(temp_path)
        observed_size = temp_path.stat().st_size
        _check_observed(source, observed_sha, observed_size)
        os.replace(temp_path, target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _check_observed(source: SourceFile, sha256: str, size_bytes: int) -> None:
    if source.sha256 and sha256 != source.sha256:
        raise DatasetError(
            f"{source.path}: SHA-256 mismatch (expected {source.sha256}, got {sha256})"
        )
    if source.size_bytes is not None and size_bytes != source.size_bytes:
        raise DatasetError(
            f"{source.path}: size mismatch (expected {source.size_bytes}, got {size_bytes})"
        )


def _check_locked(path: str, locked: dict[str, Any], sha256: str, size_bytes: int) -> None:
    expected_sha = str(locked.get("observed_sha256", ""))
    expected_size = locked.get("observed_size_bytes")
    if sha256 != expected_sha or size_bytes != expected_size:
        raise DatasetError(
            f"{path}: local snapshot differs from sources.lock.json; "
            "run `quorum packs fetch --force` to advance it explicitly"
        )


def _lock_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"invalid source lock {path}: {exc}") from exc
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        raise DatasetError(f"invalid source lock {path}: files must be a list")
    return {str(item.get("path", "")): item for item in files if item.get("path")}


def _validate_source(source: SourceFile, path: Path) -> None:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            if not next(csv.DictReader(handle), None):
                raise DatasetError(f"{source.path}: CSV contains no records")
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            if json.load(handle) is None:
                raise DatasetError(f"{source.path}: JSON is empty")
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            json.loads(next(handle))
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            json.loads(next(handle))


def _prepare_gsm8k(source_root: Path, output_root: Path) -> Path:
    train_rows = _read_jsonl(source_root / "gsm8k/train.jsonl")
    test_rows = _read_jsonl(source_root / "gsm8k/test.jsonl")
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in train_rows:
        target = validation if _bucket(row["question"], 10) == 0 else train
        target.append(_gsm8k_task(row))
    splits = {
        "train": train,
        "validation": validation,
        "promotion_test": [_gsm8k_task(row) for row in test_rows],
    }
    return _write_pack(
        output_root, "reasoning-math-gsm8k", "reasoning_math", "MIT", splits,
        [_source_manifest("gsm8k")],
        {"training_splits": ["train"], "sealed_splits": ["promotion_test"]},
    )


def _prepare_banking77(source_root: Path, output_root: Path) -> Path:
    train_rows, test_rows = _banking77_rows(source_root)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in train_rows:
        target = validation if _bucket(row["text"], 10) == 0 else train
        target.append(_banking77_task(row))
    splits = {
        "train": train,
        "validation": validation,
        "promotion_test": [_banking77_task(row) for row in test_rows],
    }
    return _write_pack(
        output_root, "extraction-banking77", "extraction_classification", "CC-BY-4.0", splits,
        [_source_manifest("banking77")],
        {"training_splits": ["train"], "sealed_splits": ["promotion_test"]},
    )


def _prepare_humaneval(source_root: Path, output_root: Path) -> Path:
    with gzip.open(source_root / "humaneval/HumanEval.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validation: list[dict[str, Any]] = []
    promotion: list[dict[str, Any]] = []
    for row in rows:
        target = promotion if _bucket(row["task_id"], 2) else validation
        target.append(_humaneval_static_task(row))
    splits = {
        "train": [dict(task) for task in _AUTHORED_CODING_TASKS],
        "validation": validation,
        "promotion_test": promotion,
    }
    return _write_pack(
        output_root, "coding-humaneval-static", "coding", "MIT", splits,
        [_source_manifest("humaneval"), _authored_source("coding")],
        {
            "training_splits": ["train"],
            "sealed_splits": ["promotion_test"],
            "evaluation_only_sources": ["humaneval"],
            "executes_untrusted_code": False,
        },
    )


def _prepare_openalex_research(source_root: Path, output_root: Path) -> Path:
    splits: dict[str, list[dict[str, Any]]] = {}
    used_ids: list[str] = []
    for split_name, (first_id, second_id, conclusion) in _RESEARCH_WORK_SPLITS.items():
        first = _read_json(source_root / f"openalex/{first_id}.json")
        second = _read_json(source_root / f"openalex/{second_id}.json")
        used_ids.extend((first_id, second_id))
        splits[split_name] = [
            _research_task(split_name, first, second, conclusion),
            _research_comparison_task(split_name, first, second),
        ]
    source = {
        "id": "openalex",
        "url": "https://api.openalex.org/works",
        "revision": "observed-2026-07-15",
        "files": [
            {
                "path": f"openalex/{work_id}.json",
                "sha256": _sha256(source_root / f"openalex/{work_id}.json"),
            }
            for work_id in used_ids
        ],
    }
    return _write_pack(
        output_root, "research-openalex", "research_synthesis", "CC0-1.0", splits,
        [source], {"training_splits": ["train"], "sealed_splits": ["promotion_test"],
                   "context_is_snapshotted": True},
    )


def _prepare_writing(output_root: Path) -> Path:
    splits = {
        split: [dict(task) for task in tasks]
        for split, tasks in _AUTHORED_WRITING_SPLITS.items()
    }
    return _write_pack(
        output_root, "writing-authored", "writing_summarization", "MIT", splits,
        [_authored_source("writing")],
        {"training_splits": ["train"], "sealed_splits": ["promotion_test"]},
    )


def _prepare_cisa_kev(source_root: Path, output_root: Path) -> Path:
    with (source_root / "cisa-kev/known_exploited_vulnerabilities.json").open(
        "r", encoding="utf-8"
    ) as handle:
        rows = json.load(handle)["vulnerabilities"]
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "promotion_test": []
    }
    for row in rows:
        bucket = _bucket(row["cveID"], 10)
        split = "promotion_test" if bucket == 0 else ("validation" if bucket == 1 else "train")
        splits[split].append(_cisa_kev_task(row))
    return _write_pack(
        output_root, "security-cisa-kev", "security_analysis", "CC0-1.0", splits,
        [_source_manifest("cisa-kev")],
        {"training_splits": ["train"], "sealed_splits": ["promotion_test"]},
    )


def _gsm8k_task(row: dict[str, Any]) -> dict[str, Any]:
    source_key = _short_hash(row["question"])
    return {
        "id": f"gsm8k-{source_key}",
        "task": f"{row['question']} End with \"Answer: <number>\".",
        "reference": row["answer"],
        "match": "numeric",
        "source_id": source_key,
        "tags": ["reasoning", "math", "word_problem"],
    }


def _banking77_task(row: dict[str, str]) -> dict[str, Any]:
    source_key = _short_hash(_normalize_text(row["text"]))
    return {
        "id": f"banking77-{source_key}",
        "task": (
            "Classify the banking support request into its fine-grained intent. "
            "Return only the snake_case intent label.\n\nRequest: " + row["text"]
        ),
        "reference": row["category"],
        "match": "exact",
        "source_id": source_key,
        "tags": ["classification", "intent"],
    }


def _humaneval_static_task(row: dict[str, Any]) -> dict[str, Any]:
    bodies = [
        row["canonical_solution"].rstrip(),
        "    pass",
        "    return None",
        "    raise NotImplementedError",
    ]
    shift = _bucket(row["task_id"], len(bodies))
    bodies = bodies[shift:] + bodies[:shift]
    answer = "ABCD"[bodies.index(row["canonical_solution"].rstrip())]
    options = "\n\n".join(f"{label})\n{body}" for label, body in zip("ABCD", bodies))
    return {
        "id": row["task_id"].replace("/", "-").lower(),
        "task": (
            "Choose the candidate Python function body that best satisfies the signature and docstring. "
            "Do not execute any candidate. End with \"Answer: <letter>\".\n\n"
            f"FUNCTION DATA:\n{row['prompt'].rstrip()}\n\nCANDIDATE BODIES:\n{options}"
        ),
        "reference": answer,
        "match": "choice",
        "source_id": row["task_id"],
        "tags": ["coding", "static", "patch_choice", "no_execution"],
    }


def _cisa_kev_task(row: dict[str, Any]) -> dict[str, Any]:
    mode = _bucket(row["cveID"], 4)
    common = (
        f"Vendor/project: {row['vendorProject']}\nProduct: {row['product']}\n"
        f"Vulnerability: {row['vulnerabilityName']}\nDescription: {row['shortDescription']}"
    )
    if mode == 0:
        task = (
            "Return the date this vulnerability was added to CISA KEV in YYYY-MM-DD format.\n\n"
            f"CVE: {row['cveID']}\n{common}"
        )
        reference = row["dateAdded"]
    elif mode == 1:
        task = (
            "Return CISA's knownRansomwareCampaignUse value for this KEV record. "
            f"Return only the value.\n\nCVE: {row['cveID']}\n{common}"
        )
        reference = row["knownRansomwareCampaignUse"]
    elif mode == 2:
        task = (
            "Return the CWE identifiers assigned to this CISA KEV record, comma-separated, "
            f"or none.\n\nCVE: {row['cveID']}\n{common}"
        )
        reference = ",".join(row.get("cwes") or []) or "none"
    else:
        task = (
            "Return the remediation due date for this CISA KEV record in YYYY-MM-DD format.\n\n"
            f"CVE: {row['cveID']}\nDate added: {row['dateAdded']}\n"
            f"Required action: {row['requiredAction']}"
        )
        reference = row["dueDate"]
    return {
        "id": f"kev-{row['cveID'].lower()}",
        "task": task,
        "reference": reference,
        "match": "exact",
        "source_id": row["cveID"],
        "tags": ["security", "cisa_kev", "evidence_extraction"],
    }


def _research_task(
    split_name: str,
    first: dict[str, Any],
    second: dict[str, Any],
    conclusion: str,
) -> dict[str, Any]:
    return {
        "id": f"openalex-{split_name}-synthesis",
        "task": (
            "Using only SOURCE A and SOURCE B, synthesize their shared design lesson in at most "
            f"120 words. End with the exact phrase \"{conclusion}\".\n\n"
            f"SOURCE A - {first['title']} ({first['publication_year']}):\n"
            f"{_abstract(first)}\n\nSOURCE B - {second['title']} ({second['publication_year']}):\n"
            f"{_abstract(second)}"
        ),
        "reference": conclusion,
        "match": "contains",
        "source_id": f"{_work_id(first)}+{_work_id(second)}",
        "tags": ["research", "synthesis", "source_grounded"],
    }


def _research_comparison_task(
    split_name: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    years = f"{first['publication_year']},{second['publication_year']}"
    return {
        "id": f"openalex-{split_name}-metadata",
        "task": (
            "Compare the two source records in one sentence, naming both publication years. "
            f"End with the exact phrase \"years: {years}\".\n\n"
            f"SOURCE A: {first['title']} - primary topic: "
            f"{(first.get('primary_topic') or {}).get('display_name', 'unknown')}\n"
            f"SOURCE B: {second['title']} - primary topic: "
            f"{(second.get('primary_topic') or {}).get('display_name', 'unknown')}"
        ),
        "reference": f"years: {years}",
        "match": "contains",
        "source_id": f"{_work_id(first)}+{_work_id(second)}",
        "tags": ["research", "comparison", "metadata"],
    }


def _write_pack(
    output_root: Path,
    pack_id: str,
    use_case: str,
    license_spdx: str,
    splits: dict[str, list[dict[str, Any]]],
    sources: list[dict[str, Any]],
    governance: dict[str, Any],
) -> Path:
    target = output_root / pack_id
    target.mkdir(parents=True, exist_ok=True)
    split_specs: dict[str, dict[str, str]] = {}
    for split_name, tasks in splits.items():
        split_path = target / f"{split_name}.json"
        payload = json.dumps({"tasks": tasks}, ensure_ascii=True, separators=(",", ":"))
        _atomic_write(split_path, payload.encode("utf-8") + b"\n")
        split_specs[split_name] = {
            "path": split_path.name,
            "fingerprint": split_fingerprint(tasks),
        }
    manifest = {
        "schema_version": 1,
        "id": pack_id,
        "version": "1.0.0",
        "use_case": use_case,
        "license": {"spdx": license_spdx, "redistribution": True, "training": True},
        "sources": sources,
        "default_metric": "deterministic_match",
        "governance": governance,
        "splits": split_specs,
    }
    _atomic_write(
        target / "manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False).encode("utf-8"),
    )
    result = verify_pack(target)
    if not result.ok:
        raise DatasetError(f"prepared {pack_id} pack is invalid: {'; '.join(result.errors)}")
    return target


def _source_manifest(source_id: str) -> dict[str, Any]:
    files = [source for source in SOURCE_FILES if source.source_id == source_id]
    return {
        "id": source_id,
        "url": files[0].url,
        "revision": files[0].revision,
        "files": [
            {"path": source.path, "sha256": source.sha256, "mutable": source.mutable}
            for source in files
        ],
    }


def _authored_source(kind: str) -> dict[str, str]:
    pack_id = "coding-humaneval-static" if kind == "coding" else "writing-authored"
    return {
        "id": f"quorum-authored-{kind}",
        "url": f"https://github.com/rinz0x0cruz/quorum/tree/main/evals/packs/{pack_id}",
        "revision": "pack-1.0.0",
    }


def _partition_smoke(
    rows: list[dict[str, Any]],
    choose_split: Any,
    transform: Any,
    limit: int,
    *,
    required: tuple[str, ...] = ("train", "validation", "promotion_test"),
) -> dict[str, list[dict[str, Any]]]:
    selected = {split: [] for split in required}
    for row in rows:
        split = choose_split(row)
        if split in selected and len(selected[split]) < limit:
            selected[split].append(transform(row))
        if all(len(tasks) == limit for tasks in selected.values()):
            break
    missing = [split for split, tasks in selected.items() if len(tasks) != limit]
    if missing:
        raise DatasetError(f"not enough records for smoke split(s): {', '.join(missing)}")
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _banking77_rows(source_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deduplicate Banking77 while preserving the official test split."""
    train_rows = _read_csv(source_root / "banking77/train.csv")
    test_rows = _dedupe_labeled_rows(_read_csv(source_root / "banking77/test.csv"))
    test_texts = {_normalize_text(row["text"]) for row in test_rows}
    train_rows = [
        row for row in _dedupe_labeled_rows(train_rows)
        if _normalize_text(row["text"]) not in test_texts
    ]
    return train_rows, test_rows


def _dedupe_labeled_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _normalize_text(row["text"])
        previous = unique.get(key)
        if previous is not None and previous["category"] != row["category"]:
            raise DatasetError(
                f"conflicting labels for normalized Banking77 text: "
                f"{previous['category']!r} and {row['category']!r}"
            )
        unique.setdefault(key, row)
    return list(unique.values())


def _read_json(path: Path) -> dict[str, Any]:
    _require(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise DatasetError(f"expected a JSON object in {path}")
    return data


def _require(path: Path) -> None:
    if not path.exists():
        raise DatasetError(f"missing source {path}; run `quorum packs fetch` first")


def _bucket(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % modulo


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _work_id(work: dict[str, Any]) -> str:
    return str(work["id"]).rsplit("/", 1)[-1]


def _abstract(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict) or not inverted:
        raise DatasetError(f"OpenAlex work {_work_id(work)} has no abstract")
    positions = [position for values in inverted.values() for position in values]
    words = [""] * (max(positions) + 1)
    for word, values in inverted.items():
        for position in values:
            words[position] = word
    return " ".join(words)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise