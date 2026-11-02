"""Large unused request bodies need not remain alive for accounting checks."""
import gc
import json
import tracemalloc
from dataclasses import asdict
from pathlib import Path
import pytest
from tokaudit.integrity import build_integrity_report, check_duplicate_spend, check_record


def row(rid='same',spend=.1,**extra):
    return {'request_id':rid,'startTime':'t','spend':spend,'prompt_tokens':10,
            'completion_tokens':2,'total_tokens':12,**extra}


def write(path,rows):
    path.write_text('\n'.join(json.dumps(r) if r is not None else '' for r in rows)+'\n')


def test_mixed_findings_and_duplicate_order_match_list_reference(tmp_path):
    rows=[row('same'),row('other',0),None,row('same',.2,total_tokens=200),
          {'id':'std','startTime':'t','response_cost':.2},row('same',.3)]
    path=tmp_path/'mixed.jsonl';write(path,rows)
    numbered=[(i+1,r) for i,r in enumerate(rows) if r is not None]
    reference=[f for line,obj in numbered for f in check_record(obj,line)]
    reference+=check_duplicate_spend(numbered)
    report=build_integrity_report(str(path))
    assert report['findings']==[asdict(f) for f in reference]
    assert report['summary']['checked_records']==5
    assert [f['line'] for f in report['findings'] if f['invariant']=='duplicate_spend_record']==[4,6]
    assert '$0.1 at line 1' in report['findings'][-1]['detail']


@pytest.mark.parametrize('first_spend',[0,-1,None,True,'bad',float('nan')])
def test_nonbillable_early_rows_do_not_claim_first_evidence(tmp_path,first_spend):
    path=tmp_path/'rows.jsonl';write(path,[row(spend=first_spend),row(spend=.2),row(spend=.3)])
    dup=[f for f in build_integrity_report(str(path))['findings'] if f['invariant']=='duplicate_spend_record']
    assert len(dup)==1 and '$0.2 at line 2' in dup[0]['detail']


def test_raw_content_retention_is_not_proportional_to_input_bytes(tmp_path):
    path=tmp_path/'large.jsonl'
    # 12 MiB of distinct unused request payload, no findings or billable ID map.
    with path.open('w') as stream:
        for i in range(192):
            stream.write(json.dumps({'id':str(i),'usageDetails':{'input':1},
                                     'input':'x'*65536+str(i)})+'\n')
    gc.collect();tracemalloc.start()
    try:
        report=build_integrity_report(str(path))
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert report['summary']['checked_records']==192
    assert report['findings']==[]
    assert peak < 3*1024*1024, f'raw accounting rows retained: peak={peak}'


def test_late_invalid_row_never_emits_partial_output(tmp_path,capsys):
    from tokaudit.cli import main
    source=tmp_path/'bad.jsonl';write(source,[row(),row(),{'messages':[]}])
    target=tmp_path/'report.json';target.write_text('keep')
    with pytest.raises(SystemExit): main(['integrity',str(source),'--format','json','--output',str(target)])
    assert target.read_text()=='keep'
    assert capsys.readouterr().out==''
