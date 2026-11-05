"""Config loading: config.yaml + TOGETHER_API_KEY from .env or environment."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["api_key"] = os.environ.get("TOGETHER_API_KEY", "")
    cfg["root"] = ROOT
    return cfg


def require_key(cfg: dict):
    if not cfg.get("api_key"):
        raise SystemExit(
            "TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
