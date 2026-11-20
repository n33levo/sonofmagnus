import hashlib

import pytest

from scripts.verify_results import ROOT,verify


def test_aggregate_rating_calculation():
    result=verify()
    assert round(result['local_estimates_recomputed']['185M'])==2853


def test_modified_evidence_is_rejected(tmp_path):
    (tmp_path/'evidence.json').write_text('{}')
    digest=hashlib.sha256(b'{}').hexdigest()
    (tmp_path/'SHA256SUMS').write_text(digest+'  evidence.json\n')
    (tmp_path/'evidence.json').write_text('{"local_elo":3000}')
    with pytest.raises(AssertionError,match='Changed evidence'):verify(tmp_path)
