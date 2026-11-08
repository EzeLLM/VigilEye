"""YAML configuration with ${ENV} interpolation and sensible built-in defaults."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "providers": {
        "openai": {
            "base_url": "${OPENAI_BASE_URL:-https://api.openai.com/v1}",
            "api_key": "${OPENAI_API_KEY:-}",
        },
        "mock": {"base_url": "mock://"},
    },
    "roles": {
        "analyst": [{"provider": "openai", "model": "gpt-4o-mini", "context_window": 128000}],
        "extractor": [{"provider": "openai", "model": "gpt-4o-mini", "context_window": 128000}],
        "vision": [{"provider": "openai", "model": "gpt-4o-mini", "context_window": 128000}],
    },
    "llm": {"timeout": 120, "max_retries": 3, "temperature": 0.2, "cache": True},
    "collectors": {"http": {"timeout": 20}},
    "analysis": {
        "lens": "general",
        "chunk_tokens": 12000,
        "max_chunks": 24,
        "extract_entities": True,
        "use_vision": False,
        "ach": False,
        "max_concurrency": 4,
    },
    "output": {"dir": "export", "formats": ["md", "json", "html"]},
    "store": {"path": ".vigileye/cases.db"},
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (python-dotenv not required). Never overrides real env."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
        return
    except Exception:
        pass
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            env = os.environ.get(m.group(1))
            if env not in (None, ""):
                return env
            return m.group(2) if m.group(2) is not None else ""
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Thin wrapper around the merged config dict."""

    def __init__(self, raw: dict | None = None, path: str | None = None, *, merge_defaults: bool = True):
        merged = deep_merge(DEFAULTS, raw or {}) if merge_defaults else (raw or {})
        # Roles/providers from the user file replace defaults wholesale when given,
        # so a file that only defines a mock provider doesn't inherit OpenAI roles.
        if raw:
            for key in ("roles", "providers"):
                if key in raw:
                    merged[key] = copy.deepcopy(raw[key])
        self.raw: dict = interpolate(merged)
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def section(self, name: str) -> dict:
        val = self.raw.get(name)
        return val if isinstance(val, dict) else {}

    def set(self, dotted: str, value: Any) -> None:
        cur = self.raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(path={self.path!r})"


def load_config(path: str | None = None) -> Config:
    _load_dotenv()
    if path is None:
        path = os.environ.get("VIGILEYE_CONFIG")
        if path is None and Path("config/vigileye.yaml").is_file():
            path = "config/vigileye.yaml"
    raw: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config {path} must be a mapping at the top level")
    cfg = Config(raw, path=path)
    if os.environ.get("VIGILEYE_STORE"):
        cfg.set("store.path", os.environ["VIGILEYE_STORE"])
    return cfg


__all__ = ["Config", "load_config", "DEFAULTS", "interpolate", "deep_merge"]
