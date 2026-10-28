"""tokaudit command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import List, Optional

from .dedup import find_duplicates
from .logs import load_log
from .prefix import simulate
from .report import build_report, to_markdown
from .rules import load_rules
from .tokenizers import get_backend
from .whatif import SCENARIOS, run_scenarios


def _cmd_analyze(args: argparse.Namespace) -> int:
    requests = load_log(args.log)
    ruleset = load_rules(args.rules)
    backend = get_backend(args.tokenizer)

    groups = defaultdict(list)
    for req in requests:
        provider = req.provider if args.provider == "auto" else args.provider
        groups[provider].append(req)

    reports = []
    for provider, group in sorted(groups.items()):
        rules = ruleset.for_provider(provider)
        sim = simulate(group, backend.encode, rules, force_family=args.policy,
                       isolate_sessions=args.isolate_sessions)
        scenarios = run_scenarios(
            args.what_if or [], group, backend.encode, rules,
            force_family=args.policy, isolate_sessions=args.isolate_sessions,
        )
        dups = find_duplicates(
            group, backend.encode,
            min_tokens=args.dup_min_tokens, shingle=max(args.dup_min_tokens // 3, 8),
        )
        reports.append(
            build_report(group, sim, dups, ruleset, rules, backend.name,
                         scenarios=scenarios)
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

    if args.format == "json":
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokaudit",
        description="Audit LLM request logs for token waste and prompt-caching savings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="analyze a JSONL request log")
    p_an.add_argument("log", help="path to a JSONL request log")
    p_an.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic", "generic"])
    p_an.add_argument("--policy", default=None,
                      help="force a policy family (e.g. legacy, gpt56_plus) for all requests")
    p_an.add_argument("--tokenizer", default="auto", choices=["auto", "exact", "estimate"])
    p_an.add_argument("--format", default="md", choices=["md", "json"])
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
    p_an.add_argument("--dup-min-tokens", type=int, default=128,
                      help="minimum size of repeated blocks to report")
    p_an.add_argument("--output", default=None, help="write the report to a file")
    p_an.set_defaults(func=_cmd_analyze)

    p_rules = sub.add_parser("rules", help="show the provider caching rules in use")
    p_rules.add_argument("--rules", default=None, help="path to a custom provider rules JSON")
    p_rules.set_defaults(func=_cmd_rules)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
