"""Provider prompt-caching rules, loaded from a versioned JSON data file.

Defaults ship in ``rules/providers.json`` with an ``as_of`` date and source
URLs, because providers change rates and minimums. Users can pass their own
file with ``--rules`` to override any value. Cost math here uses
*multipliers of the base input-token rate*, not dollars, so results stay
valid as dollar prices change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rules", "providers.json")


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


def _provider_from_dict(name: str, d: dict) -> ProviderRules:
    return ProviderRules(
        name=name,
        min_prefix_tokens=int(d.get("min_prefix_tokens", 1024)),
        ttl_seconds=int(d.get("ttl_seconds", 300)),
        write_multiplier=float(d.get("write_multiplier", 1.25)),
        read_multiplier=float(d.get("read_multiplier", 0.1)),
        refresh_on_read=bool(d.get("refresh_on_read", False)),
        enabled_by_default=bool(d.get("enabled_by_default", False)),
        max_explicit_breakpoints=int(d.get("max_explicit_breakpoints", 4)),
        model_minimums={str(k).lower(): int(v) for k, v in d.get("model_minimums", {}).items()},
        families={
            fname: FamilyRules(
                name=fname,
                min_prefix_tokens=int(fdata.get("min_prefix_tokens", d.get("min_prefix_tokens", 1024))),
                ttl_seconds=int(fdata.get("ttl_seconds", d.get("ttl_seconds", 300))),
                write_multiplier=float(fdata.get("write_multiplier", d.get("write_multiplier", 1.25))),
                read_multiplier=float(fdata.get("read_multiplier", d.get("read_multiplier", 0.1))),
                match=[str(x).lower() for x in fdata.get("match", [])],
                savings_supported=bool(fdata.get("savings_supported", True)),
                notes=[str(n) for n in fdata.get("notes", [])],
            )
            for fname, fdata in d.get("families", {}).items()
        },
        notes=[str(n) for n in d.get("notes", [])],
    )


def load_rules(path: Optional[str] = None) -> Ruleset:
    use_path = path or _DEFAULT_PATH
    with open(use_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    providers = {
        name: _provider_from_dict(name, pdata)
        for name, pdata in data.get("providers", {}).items()
    }
    if not providers:
        raise SystemExit(f"rules file {use_path} defines no providers")
    return Ruleset(
        as_of=str(data.get("as_of", "unknown")),
        sources={str(k): str(v) for k, v in data.get("sources", {}).items()},
        providers=providers,
    )
