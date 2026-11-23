from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    for key in ("output_dir", "cache_dir"):
        p = ROOT / cfg["general"][key]
        p.mkdir(parents=True, exist_ok=True)
        cfg["general"][key] = p
    return cfg
