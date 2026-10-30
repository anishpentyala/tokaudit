"""tokaudit command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from .dedup import find_duplicates
from .integrity import run_integrity
from .logs import load_log
from .prefix import simulate
from .prices import load_price_sheet
from .report import build_report, to_markdown
from .rules import load_rules
from .config import load_config
from .tokenizers import get_backend
from .whatif import SCENARIOS, run_scenarios


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    provider = args.provider if args.provider is not None else cfg.get("provider", "auto")
    tokenizer = args.tokenizer if args.tokenizer is not None else cfg.get("tokenizer", "auto")
    out_format = args.format if args.format is not None else cfg.get("format", "md")
    rules_path = args.rules if args.rules is not None else cfg.get("rules")
    policy = args.policy if args.policy is not None else cfg.get("policy")
    dup_min = args.dup_min_tokens if args.dup_min_tokens is not None else cfg.get("dup_min_tokens", 128)
    isolate = args.isolate_sessions or bool(cfg.get("isolate_sessions", False))
    what_if = args.what_if if args.what_if is not None else cfg.get("what_if")
    output = args.output if args.output is not None else cfg.get("output")
    prices_path = args.prices if args.prices is not None else cfg.get("prices")
    price_sheet = load_price_sheet(prices_path) if prices_path else None

    requests = load_log(args.log)
    ruleset = load_rules(rules_path)
    backend = get_backend(tokenizer)

    groups = defaultdict(list)
    for req in requests:
        prov = req.provider if provider == "auto" else provider
        groups[prov].append(req)

    reports = []
    for provider, group in sorted(groups.items()):
        rules = ruleset.for_provider(provider)
        sim = simulate(group, backend.encode, rules, force_family=policy,
                       isolate_sessions=isolate)
        scenarios = run_scenarios(
            what_if or [], group, backend.encode, rules,
            force_family=policy, isolate_sessions=isolate,
        )
        dups = find_duplicates(
            group, backend.encode,
            min_tokens=dup_min, shingle=max(dup_min // 3, 8),
        )
        reports.append(
            build_report(group, sim, dups, ruleset, rules, backend.name,
                         scenarios=scenarios, price_sheet=price_sheet,
                         price_source=prices_path)
        )

    if len(reports) == 1:
        combined = reports[0]
    else:
        combined = {
            "groups": reports,
            "summary": {
                "requests": sum(r["summary"]["requests"] for r in reports),
                "simulated_savings_pct": round(
                    sum(r["summary"]["simulated_savings_units"] for r in reports)
                    / max(sum(r["summary"]["baseline_units"] for r in reports), 1) * 100, 1
                ),
            },
        }

    if out_format == "json":
        out = json.dumps(combined, indent=2)
    else:
        if len(reports) == 1:
            out = to_markdown(combined)
        else:
            out = "\n---\n".join(to_markdown(r) for r in reports)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"wrote {args.output}")
    else:
        print(out)
    return 0


def _cmd_rules(args: argparse.Namespace) -> int:
    ruleset = load_rules(args.rules)
    print(f"Rules as of {ruleset.as_of}")
    for name, url in ruleset.sources.items():
        print(f"  source ({name}): {url}")
    for name, rules in sorted(ruleset.providers.items()):
        print(f"\n[{name}]")
        print(f"  min prefix tokens : {rules.min_prefix_tokens}")
        print(f"  cache TTL         : {rules.ttl_seconds}s (refresh on read: {rules.refresh_on_read})")
        print(f"  write multiplier  : {rules.write_multiplier}x base input")
        print(f"  read multiplier   : {rules.read_multiplier}x base input")
        print(f"  on by default     : {rules.enabled_by_default}")
        for note in rules.notes:
            print(f"  - {note}")
    return 0


def _cmd_integrity(args: argparse.Namespace) -> int:
    findings = run_integrity(args.log)
    by_name: Dict[str, list] = {}
    for f in findings:
        by_name.setdefault(f.invariant, []).append(f)
    lines = [f"integrity: {len(findings)} finding(s) in {args.log}"]
    for name in sorted(by_name):
        group = by_name[name]
        lines.append(f"\n{name}: {len(group)} record(s) "
                     f"- consistent with {group[0].anchor}")
        for f in group[:5]:
            lines.append(f"  line {f.line} [{f.rid}]: {f.detail}")
        if len(group) > 5:
            lines.append(f"  ... and {len(group) - 5} more")
    if findings:
        lines.append("\nA finding means 'consistent with a documented failure "
                     "class' - check the linked issue against your upstream "
                     "versions. It is a signal, not proof of misbilling. The "
                     "usage-sum check assumes the OpenAI/LiteLLM usage shape "
                     "(total = prompt + completion; detail fields are "
                     "subsets); records whose schema documents a different "
                     "sum are suppressed when the fields show it.")
    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 1 if (findings and args.fail_on_findings) else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokaudit",
        description="Audit LLM request logs for token waste and prompt-caching savings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="analyze a JSONL request log")
    p_an.add_argument("log", help="path to a JSONL request log")
    p_an.add_argument("--provider", default=None, choices=["auto", "openai", "anthropic", "generic"])
    p_an.add_argument("--policy", default=None,
                      help="force a policy family (e.g. legacy, gpt56_plus) for all requests")
    p_an.add_argument("--tokenizer", default=None, choices=["auto", "exact", "estimate"])
    p_an.add_argument("--format", default=None, choices=["md", "json"])
    p_an.add_argument("--rules", default=None, help="path to a custom provider rules JSON")
    p_an.add_argument("--what-if", action="append", choices=sorted(SCENARIOS),
                      metavar="SCENARIO",
                      help="also simulate a hypothetical scenario (repeatable): "
                           "dense = compressed timing so nothing "
                           "expires (hypothetical scenario); dedup = exact "
                           "duplicate requests "
                           "collapsed. Non-mutating; does not change the "
                           "as-is simulation.")
    p_an.add_argument("--isolate-sessions", action="store_true",
                      help="conservative scenario: also partition cache reuse by session_id "
                           "(not provider-documented behavior)")
    p_an.add_argument("--dup-min-tokens", type=int, default=None,
                      help="minimum size of repeated blocks to report")
    p_an.add_argument("--output", default=None, help="write the report to a file")
    p_an.add_argument("--prices", default=None,
                      help="path to a user-supplied price sheet JSON "
                           "({\"models\": {model: {\"input\", \"cached_input\", "
                           "\"cache_write\"}}} per 1M tokens) to also report "
                           "dollar figures; no prices ship with tokaudit")
    p_an.add_argument("--config", default=None,
                      help="path to a .tokaudit.toml config file (defaults: "
                           "./.tokaudit.toml when present; CLI flags win)")
    p_an.set_defaults(func=_cmd_analyze)

    p_int = sub.add_parser(
        "integrity",
        help="check accounting fields for internal consistency (offline; "
             "each finding is anchored to a documented upstream failure class)")
    p_int.add_argument("log", help="path to a JSONL request log")
    p_int.add_argument("--output", default=None, help="write the report to a file")
    p_int.add_argument("--fail-on-findings", action="store_true",
                       help="exit 1 when any finding fires (for CI gating)")
    p_int.set_defaults(func=_cmd_integrity)

    p_rules = sub.add_parser("rules", help="show the provider caching rules in use")
    p_rules.add_argument("--rules", default=None, help="path to a custom provider rules JSON")
    p_rules.set_defaults(func=_cmd_rules)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
