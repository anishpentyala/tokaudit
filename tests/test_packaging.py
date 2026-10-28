"""Packaging guards: installed copies of the default rules stay in sync
with the editable repo copy, and the bundled file loads."""

import filecmp
import os

from tokaudit.rules import _DEFAULT_PATH, load_rules

REPO_RULES = os.path.join(os.path.dirname(__file__), "..", "rules", "providers.json")


def test_bundled_rules_match_repo_copy():
    assert os.path.exists(REPO_RULES)
    assert filecmp.cmp(_DEFAULT_PATH, REPO_RULES, shallow=False), (
        "src/tokaudit/providers.json and rules/providers.json differ; "
        "edit the repo copy and re-bundle (cp rules/providers.json src/tokaudit/)"
    )


def test_bundled_default_rules_load():
    ruleset = load_rules()
    assert ruleset.providers and ruleset.as_of != "unknown"
