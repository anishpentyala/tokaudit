"""Optional .tokaudit.toml defaults for the analyze command.

TOML parsing uses the standard library on Python 3.11+ (tomllib) or the
``tomli`` backport when installed; otherwise a clean error is raised.
Explicit command-line flags always win over the config file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

KNOWN_KEYS = {
    "rules", "policy", "tokenizer", "provider", "format",
    "dup_min_tokens", "isolate_sessions", "what_if", "output",
}


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            raise SystemExit(
                "reading .tokaudit.toml needs Python 3.11+ (tomllib) or the "
                "'tomli' package installed"
            )
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"config file not found: {path}")
    except Exception as exc:  # TOMLDecodeError lives in the chosen module
        raise SystemExit(f"config file {path}: invalid TOML: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"config file {path}: expected a TOML table")
    return data


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return validated config values, or {} when no config applies."""
    if path is None:
        candidate = os.path.join(os.getcwd(), ".tokaudit.toml")
        if not os.path.exists(candidate):
            return {}
        path = candidate
    data = _load_toml(path)
    unknown = sorted(set(data) - KNOWN_KEYS)
    if unknown:
        raise SystemExit(
            f"config file {path}: unknown keys: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(KNOWN_KEYS))})"
        )
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key in ("dup_min_tokens",):
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                raise SystemExit(f"config file {path}: '{key}' must be an integer")
        elif key == "isolate_sessions":
            out[key] = bool(value)
        elif key == "what_if":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise SystemExit(f"config file {path}: 'what_if' must be a list of strings")
            out[key] = list(value)
        else:
            if not isinstance(value, str):
                raise SystemExit(f"config file {path}: '{key}' must be a string")
            out[key] = value
    return out
