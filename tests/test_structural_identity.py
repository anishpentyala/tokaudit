from dataclasses import replace
import pytest

from tokaudit.logs import Request, Message
from tokaudit.prefix import simulate
from tokaudit.rules import ProviderRules
from tokaudit.tokenizers import EstimatorBackend
from tokaudit.whatif import run_scenarios

ENC = EstimatorBackend().encode
RULES = ProviderRules(name='generic', min_prefix_tokens=8, ttl_seconds=300,
                      write_multiplier=1.25, read_multiplier=.1,
                      refresh_on_read=True, enabled_by_default=True)


@pytest.mark.parametrize('role', ['user', 'assistant', 'system'])
def test_literal_role_header_is_not_an_actual_message_boundary(role):
    prefix, suffix = 'left ' * 30, 'right ' * 30
    first = Request('a', 'm', 'generic', [Message('user', prefix + '\n<' + role + '>\n' + suffix)], ts=0)
    second = Request('b', 'm', 'generic', [Message('user', prefix), Message(role, suffix)], ts=10)
    assert first.canonical() == second.canonical()
    result = simulate([first, second], ENC, RULES)
    assert result.per_request[1].read == 0
    scenario, = run_scenarios(['dedup'], [first, second], ENC, RULES)
    assert any('No exact duplicate' in note for note in scenario['notes'])


@pytest.mark.parametrize('role', ['user', 'assistant', 'system'])
def test_distinct_structures_not_exact_duplicates(role):
    first = Request('a', 'm', 'generic', [Message('user', 'a\n<' + role + '>\nb')], ts=0)
    second = Request('b', 'm', 'generic', [Message('user', 'a'), Message(role, 'b')], ts=10)
    scenario, = run_scenarios(['dedup'], [first, second], ENC, RULES)
    assert any('No exact duplicate' in note for note in scenario['notes'])


def test_same_structure_still_hits_and_collapses():
    first = Request('a', 'm', 'generic', [Message('user', 'same ' * 100)], ts=0)
    second = replace(first, rid='b', ts=10)
    assert simulate([first, second], ENC, RULES).per_request[1].read > 0
    scenario, = run_scenarios(['dedup'], [first, second], ENC, RULES)
    assert any('1 exact duplicate' in note for note in scenario['notes'])


def test_true_shorter_message_prefix_still_reads():
    first = Request('a', 'm', 'generic', [Message('system', 'prefix ' * 100)], ts=0)
    second = replace(first, rid='b', ts=10, messages=first.messages + [Message('user', 'question')])
    assert simulate([first, second], ENC, RULES).per_request[1].read > 0


@pytest.mark.parametrize('reverse', [False, True])
def test_literal_boundary_alias_rejected_in_both_orders(reverse):
    a = Request('a', 'm', 'generic', [Message('user', 'left '*30 + '\n<system>\n' + 'right '*30)])
    b = Request('b', 'm', 'generic', [Message('user', 'left '*30), Message('system', 'right '*30)])
    rows = [b, a] if reverse else [a, b]
    assert simulate(rows, ENC, RULES).per_request[1].read == 0


def test_structural_identity_keeps_counts_unchanged():
    row = Request('a', 'm', 'generic', [Message('user', 'literal\n<user>\ntext')])
    total = len(ENC(row.canonical()))
    cost = simulate([row], ENC, RULES).per_request[0]
    assert cost.total_input == total
    assert cost.read + cost.written + cost.uncached == total
