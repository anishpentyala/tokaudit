"""Machine-readable integrity reports preserve coverage and complete evidence."""
import json

import pytest

from tokaudit.cli import main
from tokaudit.integrity import build_integrity_report, run_integrity


def _log(tmp_path, rows):
    path = tmp_path / 'log.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return path


def _spend(rid='r', **extra):
    row = {'request_id': rid, 'startTime': 't', 'spend': .01,
           'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
    row.update(extra)
    return row


def test_api_reports_mixed_format_counts_and_partial_coverage(tmp_path):
    path = _log(tmp_path, [
        _spend(),
        {'id': 's', 'startTime': 't', 'response_cost': .01},
        {'request_id': 'h', 'request_body': {'prompt': 'SECRET'},
         'prompt_tokens': 10},
        {'id': 'l', 'usageDetails': {'input': 10}},
        {'id': 'l2', 'usage_details': {'output': 5}},
    ])
    report = build_integrity_report(str(path))
    assert report['schema_version'] == 1
    assert report['summary'] == {
        'checked_records': 5,
        'format_counts': {'litellm_spendlogs': 1, 'litellm_standard_logging': 1,
                          'helicone': 1, 'langfuse': 2},
        'finding_count': 0, 'invariant_counts': {},
    }
    assert report['findings'] == []
    assert 'SECRET' not in json.dumps(report)
    assert 'missing optional fields are not zero' in report['limitations'][0]


def test_json_contains_all_findings_and_duplicate_cross_record_checks(tmp_path, capsys):
    path = _log(tmp_path, [_spend('same') for _ in range(9)])
    assert main(['integrity', str(path), '--format', 'json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['summary']['checked_records'] == 9
    assert report['summary']['finding_count'] == 8
    assert report['summary']['invariant_counts'] == {'duplicate_spend_record': 8}
    assert [finding['line'] for finding in report['findings']] == list(range(2, 10))
    for finding in report['findings']:
        assert set(finding) == {'invariant', 'line', 'rid', 'detail', 'anchor'}
        assert finding['rid'] == 'same'
        assert finding['anchor'].endswith('/25351')


def test_list_api_matches_report_without_changing_finding_dataclass(tmp_path):
    path = _log(tmp_path, [_spend(spend=0), _spend('r2', total_tokens=99)])
    from dataclasses import asdict
    assert [asdict(x) for x in run_integrity(str(path))] == build_integrity_report(str(path))['findings']


@pytest.mark.parametrize('report_format', ['text', 'json'])
@pytest.mark.parametrize('has_findings', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_cli_return_policy_independent_of_report_format(tmp_path, capsys, report_format, has_findings, fail):
    path = _log(tmp_path, [_spend(spend=0 if has_findings else .01)])
    output = tmp_path / 'output.txt'
    args = ['integrity', str(path), '--format', report_format, '--output', str(output)]
    if fail:
        args.append('--fail-on-findings')
    assert main(args) == int(has_findings and fail)
    rendered = capsys.readouterr().out
    assert output.read_text() == rendered
    if report_format == 'json':
        assert json.loads(rendered)['summary']['finding_count'] == int(has_findings)
    else:
        assert rendered.startswith('integrity: ')


@pytest.mark.parametrize('rows', [[], [{}], [_spend(), {'messages': []}], [42]])
def test_json_invalid_input_does_not_write_a_clean_report(tmp_path, capsys, rows):
    path = _log(tmp_path, rows)
    output = tmp_path / 'output.json'
    output.write_text('preserve')
    with pytest.raises(SystemExit):
        main(['integrity', str(path), '--format', 'json', '--output', str(output)])
    assert output.read_text() == 'preserve'
    assert capsys.readouterr().out == ''


def test_json_escapes_ids_and_never_reads_request_content(tmp_path, capsys):
    rid = 'id\n"\\|<script>'
    path = _log(tmp_path, [_spend(rid, spend=0, messages=[{'content': 'SECRET'}])])
    assert main(['integrity', str(path), '--format', 'json']) == 0
    rendered = capsys.readouterr().out
    report = json.loads(rendered)
    assert report['findings'][0]['rid'] == rid
    assert 'SECRET' not in rendered
    assert '\\n' in rendered
    assert 'may be sensitive' in report['limitations'][-1]


def test_json_findings_are_finite_even_for_nonfinite_input(tmp_path, capsys):
    path = _log(tmp_path, [_spend(spend=float('nan'), total_tokens=float('inf'))])
    assert main(['integrity', str(path), '--format', 'json']) == 0
    def reject(constant):
        raise AssertionError('JSON must not contain nonfinite literal ' + constant)
    report = json.loads(capsys.readouterr().out, parse_constant=reject)
    assert report['summary']['finding_count'] == 2
    assert all(f['invariant'] == 'non_numeric_accounting_field' for f in report['findings'])


def test_blank_lines_preserve_physical_line_numbers(tmp_path):
    path = tmp_path / 'log.jsonl'
    path.write_text('\n' + json.dumps(_spend(spend=0)) + '\n\n')
    report = build_integrity_report(str(path))
    assert report['summary']['checked_records'] == 1
    assert report['findings'][0]['line'] == 2
