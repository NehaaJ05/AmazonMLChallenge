"""Tiny known-simple end-to-end case: normalize -> block -> match -> score runs and behaves sanely."""
import pandas as pd
import config as C
from common.normalize import add_normalized
from common.scorer import score, candidate_stats
from blocking import rung0 as blocker
from matcher import rung0 as matcher


def R(rows):
    return add_normalized(pd.DataFrame(rows, columns=[C.ID, C.NAME, C.ADDR, C.COUNTRY]))


S1 = R([("s1a", "Café Blue & Co.", "12 Main St, Springfield", "US"),
        ("s1b", "Sharma Traders", "Shop 4, MG Road, Pune", "India"),
        ("s1c", "Lonely Widgets", "1 Nowhere Ln", "US")])          # singleton
OTH = R([("o1", "CAFE BLUE and CO", "12 Main St Springfield", "US"),
         ("o2", "cafe blue & co", None, "US"),                     # missing address
         ("o3", "Cafe Blue & Co", "12 Main St", "France"),         # other country -> not blocked
         ("o4", "Sharma Traders Pvt", "MG Road Pune", "India"),    # prefix block only
         ("o5", "Sharma Traders", "99 Other Rd, Delhi", "India")]) # same name, different place
GOLD = pd.DataFrame([("s1a", "o1"), ("s1a", "o2"), ("s1b", "o4")], columns=[C.S1, C.CAND])


def test_end_to_end():
    cands, _ = blocker.generate(S1, OTH)
    got = set(map(tuple, cands[[C.S1, C.CAND]].values))
    assert {("s1a", "o1"), ("s1a", "o2"), ("s1b", "o4"), ("s1b", "o5")} <= got
    assert ("s1a", "o3") not in got                                  # country blocking
    assert candidate_stats(cands, GOLD, ["s1a", "s1b", "s1c"])["candidate_recall"] == 1.0

    scored = matcher.score_pairs(matcher.attach(cands, S1, OTH))
    o2 = scored[scored[C.CAND].eq("o2")].iloc[0]
    assert pd.isna(o2["addr_sim"]) and o2["score"] == 1.0            # missing addr != mismatch
    r = score(matcher.predict(scored, 0.85), GOLD, ["s1a", "s1b", "s1c"])
    assert 0.0 <= r["f_beta"] <= 1.0 and r["singleton_correct_empty_rate"] == 1.0
