"""Rung 0 matcher: hand-weighted similarity, no learning.
  name_sim = token_sort_ratio(name_n)          (robust to word order)
  addr_sim = token_set_ratio(addr_n)           (tolerant of missing components; NB subset => 1.0, watch for false merges)
  missing address on either side => no evidence, not a mismatch (weights renormalize onto name by default)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
import config as C

NAME = "rung0_weighted_sim"
FIELDS = ["name_n", "addr_n", "country_n", "addr_missing"]


def attach(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    l = s1.set_index(C.ID)[FIELDS].add_suffix("_1")
    r = others.set_index(C.ID)[FIELDS].add_suffix("_2")
    return pairs.join(l, on=C.S1).join(r, on=C.CAND)


def _sim(a: pd.Series, b: pd.Series, scorer) -> np.ndarray:
    return cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=-1).astype(np.float32) / 100.0


def score_pairs(p: pd.DataFrame, w_name: float = 0.6, missing_addr="renorm") -> pd.DataFrame:
    p = p.copy()
    p["name_sim"] = _sim(p["name_n_1"], p["name_n_2"], fuzz.token_sort_ratio)
    p["addr_sim"] = _sim(p["addr_n_1"], p["addr_n_2"], fuzz.token_set_ratio)
    avail = ~(p["addr_missing_1"] | p["addr_missing_2"])
    p.loc[~avail, "addr_sim"] = np.nan
    w_addr = 1.0 - w_name
    if missing_addr == "renorm":
        s = np.where(avail, w_name * p["name_sim"] + w_addr * p["addr_sim"], p["name_sim"])
    else:  # numeric neutral value imputed for the missing address
        s = w_name * p["name_sim"] + w_addr * p["addr_sim"].fillna(float(missing_addr))
    p["score"] = s
    p["country_match"] = p["country_n_1"].eq(p["country_n_2"])
    return p


def predict(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # independent per-pair decisions: 0..many matches per S1, empty is a legal answer
    return scored.loc[scored["score"] >= threshold, [C.S1, C.CAND]].reset_index(drop=True)
