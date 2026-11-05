"""Provider prompt-caching rules, loaded from a versioned JSON data file.

Defaults ship inside the package (``tokaudit/providers.json``,
kept in sync with the repo copy at ``rules/providers.json``) with an
``as_of`` date and source
URLs, because providers change rates and minimums. Users can pass their own
file with ``--rules`` to override any value. Cost math here uses
*multipliers of the base input-token rate*, not dollars, so results stay
valid as dollar prices change.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .jsoninput import DuplicateJSONMember, unique_object

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "providers.json")


@dataclass
class FamilyRules:
    """Effective caching policy for one model family within a provider."""
    name: str
    min_prefix_tokens: int
    ttl_seconds: int
    write_multiplier: float
    read_multiplier: float
    match: List[str] = field(default_factory=list)
    savings_supported: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class ProviderRules:
    name: str
    min_prefix_tokens: int
    ttl_seconds: int
    write_multiplier: float
    read_multiplier: float
    refresh_on_read: bool
    enabled_by_default: bool
    max_explicit_breakpoints: int = 4
    model_minimums: Dict[str, int] = field(default_factory=dict)
    families: Dict[str, FamilyRules] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # Optional separate read lifetime. Omitted preserves legacy same-TTL rules.
    read_refresh_seconds: Optional[int] = None

    def resolve_min_prefix(self, model: str) -> int:
        m = model.lower()
        for key, value in self.model_minimums.items():
            if key in m:
                return int(value)
        return self.min_prefix_tokens

    def resolve_family(self, model: str) -> "FamilyResolution":
        """Map a model name to a policy family.

        Returns the matched family, or a resolution flagged assumed=True
        built from the provider-level defaults when no family matches.
        """
        m = model.lower()
        if not self.families and self.model_minimums:
            # Provider has a per-model minimums table (e.g. Anthropic):
            # an unrecognized model fails closed like any unknown policy,
            # because a prefix minimum alone does not verify TTL/rates.
            if not any(key in m for key in self.model_minimums):
                return FamilyResolution(
                    family=FamilyRules(
                        name=f"{self.name}-unknown",
                        min_prefix_tokens=self.min_prefix_tokens,
                        ttl_seconds=self.ttl_seconds,
                        write_multiplier=self.write_multiplier,
                        read_multiplier=self.read_multiplier,
                        savings_supported=False,
                        notes=[f"Model '{model}' is not in the {self.name} minimums table."],
                    ),
                    assumed=True,
                )
        if not self.families:
            # No families defined: the provider-level values ARE the user's
            # explicit parameters (custom rules file or constructed rules),
            # so they are supported by definition.
            return FamilyResolution(
                family=FamilyRules(
                    name=f"{self.name}-custom",
                    min_prefix_tokens=self.min_prefix_tokens,
                    ttl_seconds=self.ttl_seconds,
                    write_multiplier=self.write_multiplier,
                    read_multiplier=self.read_multiplier,
                ),
                assumed=False,
            )
        for fam in self.families.values():
            for key in fam.match:
                # Exact documented identifiers only. Dated snapshots and other
                # suffixed variants do NOT inherit rates without a documented
                # provider guarantee, so they fail closed like any unknown
                # model; the user can force a family explicitly with --policy.
                if m == key:
                    return FamilyResolution(family=fam, assumed=False)
        return FamilyResolution(
            family=FamilyRules(
                name=f"{self.name}-unknown",
                min_prefix_tokens=self.min_prefix_tokens,
                ttl_seconds=self.ttl_seconds,
                write_multiplier=self.write_multiplier,
                read_multiplier=self.read_multiplier,
                savings_supported=False,
                notes=[f"No policy family matched model '{model}'."],
            ),
            assumed=True,
        )


@dataclass
class FamilyResolution:
    family: FamilyRules
    assumed: bool


@dataclass
class Ruleset:
    as_of: str
    sources: Dict[str, str]
    providers: Dict[str, ProviderRules]

    def for_provider(self, name: str) -> ProviderRules:
        if name == "generic":
            name = "openai"
        if name not in self.providers:
            raise SystemExit(f"no caching rules for provider '{name}'; pass --rules with your own")
        return self.providers[name]


def _count(value, label):
    # Keep integral floats and decimal integer strings used by older custom
    # policies, but never truncate fractions or treat booleans as counts.
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative integer")
    if isinstance(value, str):
        if not re.fullmatch(r"[+-]?[0-9]+", value.strip()):
            raise ValueError(f"{label} must be a finite non-negative integer")
    elif not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative integer")
    try:
        number = float(value)
        count = int(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{label} must be a finite non-negative integer") from None
    if count < 0:
        raise ValueError(f"{label} must be non-negative")
    if not math.isfinite(number) or (not isinstance(value, str) and value != count):
        raise ValueError(f"{label} must be a finite non-negative integer")
    return count


def _multiplier(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{label} must be finite and non-negative") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _flag(value, label):
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _strings(value, label):
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return list(value)


def _provider_from_dict(name: str, d: dict) -> ProviderRules:
    counts = {key: _count(d.get(key, default), key) for key, default in (
        ("min_prefix_tokens", 1024), ("ttl_seconds", 300), ("max_explicit_breakpoints", 4)
    )}
    multipliers = {key: _multiplier(d.get(key, default), key) for key, default in (
        ("write_multiplier", 1.25), ("read_multiplier", 0.1)
    )}
    minimums = d.get("model_minimums", {})
    if not isinstance(minimums, dict):
        raise ValueError("model_minimums must be an object")
    families = d.get("families", {})
    if not isinstance(families, dict):
        raise ValueError("families must be an object")
    parsed_families = {}
    for fname, fdata in families.items():
        label = f"family '{fname}'"
        if not isinstance(fdata, dict):
            raise ValueError(f"{label} must be an object")
        parsed_families[fname] = FamilyRules(
            name=fname,
            min_prefix_tokens=_count(fdata.get("min_prefix_tokens", counts["min_prefix_tokens"]),
                                     f"{label}: min_prefix_tokens"),
            ttl_seconds=_count(fdata.get("ttl_seconds", counts["ttl_seconds"]), f"{label}: ttl_seconds"),
            write_multiplier=_multiplier(fdata.get("write_multiplier", multipliers["write_multiplier"]),
                                         f"{label}: write_multiplier"),
            read_multiplier=_multiplier(fdata.get("read_multiplier", multipliers["read_multiplier"]),
                                        f"{label}: read_multiplier"),
            match=[x.lower() for x in _strings(fdata.get("match", []), f"{label}: match")],
            savings_supported=_flag(fdata.get("savings_supported", True), f"{label}: savings_supported"),
            notes=_strings(fdata.get("notes", []), f"{label}: notes"),
        )
    refresh = d.get("read_refresh_seconds")
    return ProviderRules(
        name=name, **counts, **multipliers,
        refresh_on_read=_flag(d.get("refresh_on_read", False), "refresh_on_read"),
        enabled_by_default=_flag(d.get("enabled_by_default", False), "enabled_by_default"),
        read_refresh_seconds=None if refresh is None else _count(refresh, "read_refresh_seconds"),
        model_minimums={k.lower(): _count(v, "model_minimums") for k, v in minimums.items()},
        families=parsed_families,
        notes=_strings(d.get("notes", []), "notes"),
    )


def load_rules(path: Optional[str] = None) -> Ruleset:
    use_path = path or _DEFAULT_PATH
    try:
        with open(use_path, "r", encoding="utf-8") as fh:
            data = json.load(fh, object_pairs_hook=unique_object)
    except FileNotFoundError:
        raise SystemExit(f"rules file not found: {use_path}")
    except DuplicateJSONMember:
        raise SystemExit(f"rules file {use_path}: duplicate JSON object member") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"rules file {use_path}: invalid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("providers"), dict):
        raise SystemExit(
            f"rules file {use_path}: expected an object with a 'providers' object"
        )
    providers = {}
    for name, pdata in data["providers"].items():
        if not isinstance(pdata, dict):
            raise SystemExit(f"rules file {use_path}: provider '{name}' must be an object")
        try:
            rules = _provider_from_dict(name, pdata)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"rules file {use_path}: provider '{name}': {exc}")
        providers[name] = rules
    if not providers:
        raise SystemExit(f"rules file {use_path} defines no providers")
    sources = data.get("sources", {})
    if not isinstance(sources, dict) or not all(isinstance(v, str) for v in sources.values()):
        raise SystemExit(f"rules file {use_path}: sources must be an object of strings")
    return Ruleset(
        as_of=str(data.get("as_of", "unknown")),
        sources=dict(sources),
        providers=providers,
    )
