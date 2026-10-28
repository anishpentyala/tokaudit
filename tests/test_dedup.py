from tokaudit.dedup import find_duplicates
from tokaudit.logs import Message, Request
from tokaudit.tokenizers import EstimatorBackend

ENC = EstimatorBackend().encode
BLOCK = "This is a long repeated policy document paragraph. " * 40  # ~500 estimator tokens


def req(rid, text):
    return Request(rid=rid, model="m", provider="openai", messages=[Message("user", text)])


def test_detects_repeated_block():
    reqs = [req(f"r{i}", f"intro {i}\n{BLOCK}\nquestion {i}") for i in range(3)]
    dups = find_duplicates(reqs, ENC, min_tokens=100, shingle=20)
    assert dups, "expected the repeated block to be detected"
    top = dups[0]
    assert top.token_count >= 100
    assert len(top.occurrences) >= 2
    assert top.repeated_tokens >= top.token_count


def test_no_duplicates_in_distinct_requests():
    reqs = [req(f"r{i}", f"completely different content {i} " + (str(i) * 300)) for i in range(3)]
    dups = find_duplicates(reqs, ENC, min_tokens=100, shingle=20)
    assert dups == []


def test_respects_min_size():
    small = "echo echo echo " * 5
    reqs = [req(f"r{i}", small) for i in range(3)]
    assert find_duplicates(reqs, ENC, min_tokens=100, shingle=20) == []
