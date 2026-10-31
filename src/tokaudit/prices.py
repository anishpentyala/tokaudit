"""User-supplied price sheets: convert simulated units into dollars.

No prices ship with tokaudit; a sheet is user data. Sheets fail closed:
unknown keys, missing rates, negative numbers and non-numeric values are
all rejected. Requests whose model is absent from the sheet are excluded
from dollar figures (their units are unaffected).
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional

RATE_KEYS = ("input", "cached_input", "cache_write")
TOP_KEYS = {"models", "as_of", "note"}


def load_price_sheet(path: str) -> Dict[str, Dict[str, float]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"price sheet not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"price sheet {path}: invalid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise SystemExit(
            f"price sheet {path}: expected an object with a 'models' object"
        )
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        raise SystemExit(
            f"price sheet {path}: unknown keys: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(TOP_KEYS))})"
        )
    out: Dict[str, Dict[str, float]] = {}
    for model, rates in data["models"].items():
        if not isinstance(rates, dict):
            raise SystemExit(f"price sheet {path}: '{model}' must be an object")
        unknown_r = sorted(set(rates) - set(RATE_KEYS))
        if unknown_r:
            raise SystemExit(
                f"price sheet {path}: '{model}' has unknown rate keys: "
                f"{', '.join(unknown_r)} (known: {', '.join(RATE_KEYS)})"
            )
        missing = [k for k in RATE_KEYS if k not in rates]
        if missing:
            raise SystemExit(
                f"price sheet {path}: '{model}' is missing rates: {', '.join(missing)}"
            )
        entry = {}
        for k in RATE_KEYS:
            try:
                if isinstance(rates[k], bool):
                    raise ValueError("boolean rate")
                v = float(rates[k])
            except (TypeError, ValueError, OverflowError):
                raise SystemExit(
                    f"price sheet {path}: '{model}.{k}' must be a number"
                )
            if not math.isfinite(v):
                raise SystemExit(
                    f"price sheet {path}: '{model}.{k}' must be finite"
                )
            if v < 0:
                raise SystemExit(
                    f"price sheet {path}: '{model}.{k}' must be non-negative"
                )
            entry[k] = v
        out[str(model)] = entry
    if not out:
        raise SystemExit(f"price sheet {path}: 'models' is empty")
    return out


def cost_usd(uncached: int, read: int, written: int,
             rates: Dict[str, float]) -> float:
    """Dollars for one request's token flows, rates per 1M tokens."""
    return (
        uncached * rates["input"]
        + read * rates["cached_input"]
        + written * rates["cache_write"]
    ) / 1_000_000.0
