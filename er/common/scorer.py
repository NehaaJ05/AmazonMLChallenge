"""The ONE shared scorer. Per-S1-entity F_beta, macro-averaged over the evaluated S1 set.
  gold empty & pred empty -> 1.0 ; exactly one empty -> 0.0 ; else standard F_beta.
Uses F_beta = (1+b^2)*TP / (b^2*|gold| + |pred|), algebraically identical, and it handles the edge cases.
TODO(both): diff against the official metric definition / reference script before trusting it."""
import numpy as np
import pandas as pd
import config as C


def _ids(eval_ids) -> pd.Index:
    return pd.Index(pd.unique(pd.Series(list(eval_ids), dtype=object)), name=C.S1)


def _prep(pairs, ids):
    p = pairs[[C.S1, C.CAND]].drop_duplicates()
    return p[p[C.S1].isin(ids)]


def score(pred: pd.DataFrame, gold: pd.DataFrame, eval_ids, beta: float = C.BETA) -> dict:
    ids = _ids(eval_ids)
    pred, gold = _prep(pred, ids), _prep(gold, ids)
    n_pred = pred.groupby(C.S1).size().reindex(ids, fill_value=0)
    n_gold = gold.groupby(C.S1).size().reindex(ids, fill_value=0)
    tp = pred.merge(gold, on=[C.S1, C.CAND]).groupby(C.S1).size().reindex(ids, fill_value=0)

    b2 = beta ** 2
    denom = (b2 * n_gold + n_pred).astype(float)
    f = pd.Series(np.where(denom == 0, 1.0, (1 + b2) * tp / denom.replace(0, 1)), index=ids)

    single = n_gold.eq(0)
    TP, NP, NG = int(tp.sum()), int(n_pred.sum()), int(n_gold.sum())
    nan = float("nan")
    return {
        "f_beta": float(f.mean()),
        "f_beta_singletons": float(f[single].mean()) if single.any() else nan,
        "f_beta_nonsingletons": float(f[~single].mean()) if (~single).any() else nan,
        "n_entities": len(ids),
        "n_singletons": int(single.sum()),
        "singleton_correct_empty_rate": float(n_pred[single].eq(0).mean()) if single.any() else nan,
        "nonsingleton_empty_pred_rate": float(n_pred[~single].eq(0).mean()) if (~single).any() else nan,
        "pair_precision": TP / NP if NP else nan,
        "pair_recall": TP / NG if NG else nan,
        "n_pred_pairs": NP,
    }


def candidate_stats(cands: pd.DataFrame, gold: pd.DataFrame, eval_ids) -> dict:
    ids = _ids(eval_ids)
    cands, gold = _prep(cands, ids), _prep(gold, ids)
    hit = gold.merge(cands, on=[C.S1, C.CAND], how="left", indicator=True)["_merge"].eq("both").to_numpy()
    per = cands.groupby(C.S1).size().reindex(ids, fill_value=0)
    covered = gold.assign(hit=hit).groupby(C.S1)["hit"].all()
    oracle = score(gold[hit], gold, ids)  # what a PERFECT matcher would score on these candidates
    nan = float("nan")
    return {
        "candidate_recall": float(hit.mean()) if len(gold) else nan,
        "entity_full_coverage": float(covered.mean()) if len(covered) else nan,
        "cands_per_s1_mean": float(per.mean()),
        "cands_per_s1_p50": float(per.median()),
        "cands_per_s1_p99": float(per.quantile(0.99)),
        "cands_per_s1_max": int(per.max()) if len(per) else 0,
        "n_candidate_pairs": int(len(cands)),
        "oracle_f_beta": oracle["f_beta"],
    }
