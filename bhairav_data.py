"""
Load runtime configuration from data/*.json and prompts/*.txt (no hardcoded dictionaries in code).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import BASE_DIR, DATA_DIR

PROMPTS_DIR = BASE_DIR / "prompts"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    data = _read_json(DATA_DIR / "stopwords.json", {"words": []})
    return frozenset(w.lower() for w in data.get("words", []))


@lru_cache(maxsize=1)
def get_domain_signals() -> dict[str, list[str]]:
    return _read_json(DATA_DIR / "domain_signals.json", {})


@lru_cache(maxsize=1)
def get_chitchat_config() -> dict:
    return _read_json(
        DATA_DIR / "chitchat.json",
        {"responses": {}, "patterns": {}},
    )


@lru_cache(maxsize=1)
def get_intent_patterns() -> dict:
    return _read_json(DATA_DIR / "intent_patterns.json", {})


@lru_cache(maxsize=1)
def get_routing_policies() -> dict:
    return _read_json(DATA_DIR / "routing_policies.json", {})


@lru_cache(maxsize=1)
def get_query_variants() -> dict:
    return _read_json(DATA_DIR / "query_variants.json", {})


def load_prompt(name: str, **fmt) -> str:
    """Load prompts/<name>.txt; optional .format(**fmt)."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    return text.format(**fmt) if fmt else text


def load_prompt_optional(name: str, fallback: str, **fmt) -> str:
    try:
        return load_prompt(name, **fmt)
    except FileNotFoundError:
        return fallback.format(**fmt) if fmt else fallback
