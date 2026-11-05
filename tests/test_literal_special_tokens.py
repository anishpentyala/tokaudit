"""Provider-looking delimiter text in exported content is ordinary input."""
import json

import pytest

from tokaudit.cli import main
from tokaudit.tokenizers import TiktokenBackend,_BoundedEncoder


@pytest.mark.parametrize('marker',['<|endoftext|>','<|endofprompt|>'])
@pytest.mark.parametrize('text', ['{}', 'Explain this delimiter: {}. Keep it as text.', '{} {}'])
def test_exact_backend_counts_literal_reserved_strings(marker,text):
    backend=TiktokenBackend()
    value=text.format(marker,marker)
    expected=backend._enc.encode(value,disallowed_special=())
    assert backend.encode(value)==expected
    assert backend.count(value)==len(expected)
    assert backend._enc.decode(expected)==value


def test_bounded_encoder_preserves_literal_content_and_copies():
    backend=TiktokenBackend();cached=_BoundedEncoder(backend.encode)
    text='quoted <|endoftext|> in source'
    expected=backend._enc.encode(text,disallowed_special=())
    first=cached(text);first.append(-1)
    assert cached(text)==expected


@pytest.mark.parametrize('mode',['exact','auto'])
def test_cli_can_audit_literal_marker_in_logs(tmp_path,capsys,mode):
    path=tmp_path/'log.jsonl'
    text='policy '*2000+'quoted <|endoftext|> not a control token'
    path.write_text('\n'.join(json.dumps({'id':str(i),'model':'gpt-6-astra','prompt':text}) for i in range(2)))
    assert main(['analyze',str(path),'--tokenizer',mode,'--format','json'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['summary']['requests_simulated']==2
    assert result['summary']['tokenizer']=='exact'
    assert len(result['duplicate_blocks'])==1


def test_ordinary_text_encoding_unchanged():
    backend=TiktokenBackend();text='A normal plain-text request.'
    assert backend.encode(text)==backend._enc.encode(text)
