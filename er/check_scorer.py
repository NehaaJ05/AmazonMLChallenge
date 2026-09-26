"""Scorer sign-off on REAL ground truth. Each line prints expected vs got; all must say OK.
  python check_scorer.py"""
import pandas as pd
import config as C
from common import io
from common.scorer import score

gold = io.load_ground_truth()
s1_ids = pd.read_csv(C.TRAIN["S1"], sep="\t", dtype=str, usecols=[C.ID])[C.ID]
single_rate = 1 - gold[C.S1].nunique() / len(s1_ids)
first_only = gold.drop_duplicates(C.S1)                      # 1 correct match per non-singleton, P=1
r_first = first_only.merge(gold.groupby(C.S1).size().rename("g"), left_on=C.S1, right_index=True)
exp_first = single_rate + ((1.25 / (0.25 * r_first["g"] + 1)).sum()) / len(s1_ids)

for name, pred, exp in [
    ("perfect prediction", gold, 1.0),
    ("predict nothing (= singleton rate)", gold.iloc[:0], single_rate),
    ("first gold match only", first_only, exp_first),
]:
    got = score(pred, gold, s1_ids)["f_beta"]
    print(f"{'OK ' if abs(got-exp) < 1e-9 else 'BAD'} {name:38s} expected {exp:.6f}  got {got:.6f}")
print(f"S1={len(s1_ids):,}  gold pairs={len(gold):,}  mean matches/S1={len(gold)/len(s1_ids):.3f} (audit: 3.46)")
