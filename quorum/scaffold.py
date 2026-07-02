"""``quorum init`` -- scaffold a working config and data directory.

Non-destructive: never overwrites an existing config. Prints the next steps a
fresh user should take on their machine.
"""
from __future__ import annotations

import os
import shutil

from .config import DEFAULT_CONFIG, load_config


def run(args) -> int:
    created = []

    if not any(os.path.exists(c) for c in ("config.yaml", "config.yml", "config.json")):
        if os.path.exists("config.example.yaml"):
            shutil.copyfile("config.example.yaml", "config.yaml")
        else:
            _write_yaml_defaults("config.yaml")
        created.append("config.yaml")

    cfg = load_config(getattr(args, "config", None))

    data_dir = os.path.dirname(cfg["output"]["db_path"]) or "."
    if data_dir and not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        created.append(data_dir + os.sep)

    if os.path.exists(".env.example") and not os.path.exists(".env"):
        shutil.copyfile(".env.example", ".env")
        created.append(".env")

    print("  created: " + ", ".join(created) if created else "  already initialized (nothing to do)")

    print("\nNext steps:")
    print("  1. Edit config.yaml           # pick your council models + providers")
    print("  2. Set a provider key         # e.g. QUORUM_OPENROUTER_KEY in .env (never in config)")
    print("  3. quorum models --ping       # confirm the endpoints are reachable")
    print("  4. quorum run \"<your task>\"    # deliberate to a good-enough answer")
    print("  5. quorum selftest            # offline sanity check (no key needed)")
    return 0


def _write_yaml_defaults(path: str) -> None:
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=False, default_flow_style=False)
    except ImportError:
        import json
        with open(path.replace(".yaml", ".json"), "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=2)
