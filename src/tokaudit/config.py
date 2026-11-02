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
    "dup_min_tokens", "isolate_sessions", "what_if", "output", "prices",
    "omit_content_previews",
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
        if key == "dup_min_tokens":
            if type(value) is not int:
                raise SystemExit(f"config file {path}: '{key}' must be an integer")
            out[key] = value
        elif key in ("isolate_sessions", "omit_content_previews"):
            if type(value) is not bool:
                raise SystemExit(f"config file {path}: '{key}' must be boolean")
            out[key] = value
        elif key == "what_if":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise SystemExit(f"config file {path}: 'what_if' must be a list of strings")
            from .whatif import SCENARIOS
            unknown_scenarios = sorted(set(value) - set(SCENARIOS))
            if unknown_scenarios:
                raise SystemExit(f"config file {path}: unknown what_if scenarios: "
                                 + ", ".join(unknown_scenarios))
            out[key] = list(value)
        else:
            if not isinstance(value, str):
                raise SystemExit(f"config file {path}: '{key}' must be a string")
            choices = {
                "provider": ("auto", "openai", "anthropic", "generic"),
                "tokenizer": ("auto", "exact", "estimate"),
                "format": ("md", "json"),
            }
            if key in choices and value not in choices[key]:
                raise SystemExit(f"config file {path}: invalid '{key}' value {value!r}; "
                                 f"choose from {', '.join(choices[key])}")
            out[key] = value
    return out
