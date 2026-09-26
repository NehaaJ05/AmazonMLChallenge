import pandas as pd
import pytest
import config as C
from common import submission
from common.scorer import score


def P(rows):
    return pd.DataFrame(rows, columns=[C.S1, C.CAND])


def test_readme_example_roundtrip(tmp_path):
    ids = ["S1-00001", "S1-00002", "S1-00003"]
    pairs = P([("S1-00001", "S3-00812"), ("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"),
               ("S1-00001", "S2-00047"), ("S1-00002", "S3-00004")])            # dup + unsorted on purpose
    f = tmp_path / "m.tsv"
    submission.write(pairs, ids, f, C.SUB_MATCH)
    assert f.read_bytes().decode() == ("source1_entity_id\tmatched_entity_ids\n"
                                       "S1-00001\tS2-00047,S2-00193,S3-00812\n"
                                       "S1-00002\tS3-00004\n"
                                       "S1-00003\t\n")                       # README example, byte for byte
    back = submission.read(f, C.SUB_MATCH)
    assert set(map(tuple, back.values)) == set(map(tuple, pairs.drop_duplicates().values))


def test_readme_worked_score_example():
    pred = P([("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"), ("S1-00001", "S3-00812")])
    gold = P([("S1-00001", "S2-00047"), ("S1-00001", "S3-00812")])
    assert score(pred, gold, ["S1-00001"])["f_beta"] == pytest.approx(0.714, abs=5e-4)


def test_check_catches_problems(tmp_path):
    ids, others = ["S1-1", "S1-2"], ["S2-1", "S3-1"]
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    submission.write(P([("S1-1", "S2-1")]), ids, m, C.SUB_MATCH)
    submission.write(P([("S1-1", "S2-1"), ("S1-1", "S3-1")]), ids, c, C.SUB_CAND)
    assert submission.check(m, c, ids, others) == []
    submission.write(P([("S1-1", "S3-1"), ("S1-2", "S2-9")]), ids, m, C.SUB_MATCH)
    issues = submission.check(m, c, ids, others)
    assert any("not in test" in s for s in issues) and any("subset" in s for s in issues)
