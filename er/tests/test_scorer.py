"""Hand-computed cases for the shared scorer. Must pass before any number goes in the log."""
import math
import pandas as pd
import pytest
import config as C
from common.scorer import score, candidate_stats


def P(rows):
    return pd.DataFrame(rows, columns=[C.S1, C.CAND])


GOLD = P([("a", "x"), ("a", "y"), ("b", "x"), ("e", "x")])  # c, d are singletons
IDS = ["a", "b", "c", "d", "e"]


def test_edge_cases_and_partial():
    pred = P([("a", "x"), ("a", "z"),                 # P=.5 R=.5 -> .5
              ("b", "x"), ("b", "y"), ("b", "z"),     # P=1/3 R=1 -> 1.25/3.25
              ("d", "q"),                             # singleton, predicted -> 0
              ("zzz", "x")])                          # not in eval set -> ignored
    r = score(pred, GOLD, IDS)                        # c: empty/empty -> 1 ; e: missed -> 0
    exp = {"a": .5, "b": 1.25 / 3.25, "c": 1.0, "d": 0.0, "e": 0.0}
    assert r["f_beta"] == pytest.approx(sum(exp.values()) / 5)
    assert r["f_beta_singletons"] == pytest.approx(0.5)
    assert r["singleton_correct_empty_rate"] == pytest.approx(0.5)
    assert r["pair_precision"] == pytest.approx(2 / 6)
    assert r["pair_recall"] == pytest.approx(2 / 4)


def test_perfect_and_duplicates():
    assert score(pd.concat([GOLD, GOLD]), GOLD, IDS)["f_beta"] == pytest.approx(1.0)


def test_empty_prediction():
    r = score(P([]), GOLD, IDS)
    assert r["f_beta"] == pytest.approx(2 / 5)       # only the two singletons score
    assert math.isnan(r["pair_precision"])


def test_beta_weights_precision():
    # same TP, one FP vs one FN: F0.5 must prefer the FN (precision-heavy)
    g = P([("a", "x"), ("a", "y")])
    fp = score(P([("a", "x"), ("a", "y"), ("a", "z")]), g, ["a"])["f_beta"]
    fn = score(P([("a", "x")]), g, ["a"])["f_beta"]
    assert fn > fp


def test_candidate_stats():
    cands = P([("a", "x"), ("a", "q"), ("b", "x"), ("c", "r")])
    s = candidate_stats(cands, GOLD, IDS)
    assert s["candidate_recall"] == pytest.approx(2 / 4)
    assert s["entity_full_coverage"] == pytest.approx(1 / 3)   # a partial, b full, e none
    assert s["cands_per_s1_mean"] == pytest.approx(4 / 5)
    # oracle: a -> P=1,R=.5 ; b ->1 ; c,d ->1 ; e ->0
    assert s["oracle_f_beta"] == pytest.approx((1.25 * .5 / (.25 + .5) + 1 + 1 + 1 + 0) / 5)
