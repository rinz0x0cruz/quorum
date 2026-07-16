"""Load and verify versioned evaluation packs."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SPLITS = ("train", "validation", "promotion_test")
SUPPORTED_MATCHES = {"numeric", "choice", "boolean", "exact", "contains", "regex"}
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class PackVerification:
    """Result of checking one evaluation pack."""

    pack_id: str
    version: str
    fingerprints: dict[str, str]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether the pack passed every required check."""
        return not self.errors


class PackError(ValueError):
    """Raised when an evaluation pack cannot be loaded safely."""


def split_fingerprint(tasks: list[dict[str, Any]]) -> str:
    """Return a canonical SHA-256 fingerprint for an ordered task split."""
    canonical = [_canonical_task(task) for task in tasks]
    payload = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_pack(path: str | Path) -> PackVerification:
    """Verify manifest metadata, split fingerprints, and split isolation."""
    root, manifest_path = _resolve_manifest(path)
    errors: list[str] = []
    try:
        manifest = _read_mapping(manifest_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return PackVerification("", "", {}, (f"manifest: {exc}",))

    pack_id = str(manifest.get("id", "")).strip()
    version = str(manifest.get("version", "")).strip()
    if not pack_id:
        errors.append("manifest.id is required")
    if not _SEMVER_RE.fullmatch(version):
        errors.append("manifest.version must be semantic versioning (for example, 1.0.0)")
    if not str(manifest.get("use_case", "")).strip():
        errors.append("manifest.use_case is required")

    license_info = manifest.get("license")
    if not isinstance(license_info, dict) or not str(license_info.get("spdx", "")).strip():
        errors.append("manifest.license.spdx is required")
    else:
        for permission in ("redistribution", "training"):
            if not isinstance(license_info.get(permission), bool):
                errors.append(f"manifest.license.{permission} must be true or false")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("manifest.sources must contain at least one attributed source")
    else:
        for index, source in enumerate(sources, 1):
            if not isinstance(source, dict):
                errors.append(f"manifest.sources[{index}] must be a mapping")
                continue
            for field in ("id", "url", "revision"):
                if not str(source.get(field, "")).strip():
                    errors.append(f"manifest.sources[{index}].{field} is required")

    split_specs = manifest.get("splits")
    if not isinstance(split_specs, dict):
        split_specs = {}
        errors.append("manifest.splits must be a mapping")

    tasks_by_split: dict[str, list[dict[str, Any]]] = {}
    fingerprints: dict[str, str] = {}
    for split_name in REQUIRED_SPLITS:
        spec = split_specs.get(split_name)
        if not isinstance(spec, dict):
            errors.append(f"manifest.splits.{split_name} is required")
            continue
        relative_path = str(spec.get("path", "")).strip()
        expected_fingerprint = str(spec.get("fingerprint", "")).strip()
        if not relative_path:
            errors.append(f"manifest.splits.{split_name}.path is required")
            continue
        split_path = root / relative_path
        try:
            tasks = _read_tasks(split_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"{split_name}: {exc}")
            continue
        tasks_by_split[split_name] = tasks
        fingerprint = split_fingerprint(tasks)
        fingerprints[split_name] = fingerprint
        if not expected_fingerprint:
            errors.append(f"manifest.splits.{split_name}.fingerprint is required")
        elif expected_fingerprint != fingerprint:
            errors.append(
                f"{split_name}: fingerprint mismatch (expected {expected_fingerprint}, got {fingerprint})"
            )
        errors.extend(_task_errors(split_name, tasks))

    seen_ids: dict[str, str] = {}
    seen_tasks: dict[str, tuple[str, str]] = {}
    for split_name, tasks in tasks_by_split.items():
        for task in tasks:
            task_id = str(task.get("id", "")).strip()
            previous_split = seen_ids.get(task_id)
            if task_id and previous_split:
                errors.append(f"task id {task_id!r} appears in both {previous_split} and {split_name}")
            elif task_id:
                seen_ids[task_id] = split_name

            prompt_key = _normalize_text(str(task.get("task", "")))
            previous = seen_tasks.get(prompt_key)
            if prompt_key and previous and previous[0] != split_name:
                errors.append(
                    f"task text for {task_id!r} duplicates {previous[1]!r} across "
                    f"{previous[0]} and {split_name}"
                )
            elif prompt_key:
                seen_tasks[prompt_key] = (split_name, task_id)

    return PackVerification(pack_id, version, fingerprints, tuple(errors))


def load_split(path: str | Path, split: str) -> list[dict[str, Any]]:
    """Load one verified split from an evaluation pack."""
    if split not in REQUIRED_SPLITS:
        raise PackError(f"unknown split {split!r}; expected one of {', '.join(REQUIRED_SPLITS)}")
    result = verify_pack(path)
    if not result.ok:
        raise PackError("; ".join(result.errors))
    root, manifest_path = _resolve_manifest(path)
    manifest = _read_mapping(manifest_path)
    return _read_tasks(root / manifest["splits"][split]["path"])


def _resolve_manifest(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate, candidate / "manifest.yaml"
    return candidate.parent, candidate


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return data


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) if path.suffix == ".json" else yaml.safe_load(handle)
    if isinstance(data, dict):
        data = data.get("tasks")
    if not isinstance(data, list):
        raise ValueError(f"expected a task list in {path}")
    if not all(isinstance(task, dict) for task in data):
        raise ValueError(f"every task in {path} must be a mapping")
    return data


def _task_errors(split_name: str, tasks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    local_ids: set[str] = set()
    for index, task in enumerate(tasks, 1):
        task_id = str(task.get("id", "")).strip()
        label = task_id or f"item {index}"
        if not task_id:
            errors.append(f"{split_name} item {index}: id is required")
        elif task_id in local_ids:
            errors.append(f"{split_name}: duplicate task id {task_id!r}")
        local_ids.add(task_id)
        if not str(task.get("task", "")).strip():
            errors.append(f"{split_name} {label}: task is required")
        reference = task.get("reference") or task.get("expected") or task.get("answer")
        if reference is None and not str(task.get("rubric", "")).strip():
            errors.append(f"{split_name} {label}: reference or rubric is required")
        match = task.get("match")
        if match is not None and match not in SUPPORTED_MATCHES:
            errors.append(f"{split_name} {label}: unsupported match type {match!r}")
    return errors


def _canonical_task(task: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(task)
    expected = canonical.pop("expected", None)
    answer = canonical.pop("answer", None)
    reference = expected if expected is not None else answer
    if canonical.get("reference") is None and reference is not None:
        canonical["reference"] = reference
    return canonical


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())