"""Rung 0 end-to-end on the frozen validation split.
  python run_rung0.py --owner A+B --notes "first clean run"
Outputs: outputs/rung0/{candidate_pairs_val.tsv, matching_results_val.tsv, scored_val.parquet, sweep.tsv}
and one appended row in experiments/log.tsv."""
import argparse, json, time
import numpy as np
import pandas as pd
import config as C
from common import io, normalize, split, scorer, explog
from blocking import rung0 as blocker
from matcher import rung0 as matcher


def load_normalized(files, cache):
    if cache.exists():
        return pd.read_parquet(cache)
    df = normalize.add_normalized(io.load_sources(files))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--max-block", type=int, default=500)
    ap.add_argument("--w-name", type=float, default=0.6)
    ap.add_argument("--missing-addr", default="renorm")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--owner", default="A+B")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    out = C.OUT / "rung0"
    t0 = time.time()

    df = load_normalized(C.TRAIN, C.OUT / "cache" / "train_norm_rung0.parquet")
    gold = io.load_ground_truth()
    s1_all = df[df["source"].eq("S1")]
    val_ids = split.get_or_make_val_ids(s1_all[C.ID])
    s1 = s1_all[s1_all[C.ID].isin(val_ids)]
    others = df[df["source"].ne("S1")]
    print(f"[load] {len(df):,} records, val S1 = {len(s1):,}  ({time.time()-t0:.0f}s)")

    cands, bstats = blocker.generate(s1, others, args.prefix_len, args.max_block)
    io.write_pairs(cands, out / "candidate_pairs_val.tsv", extra_cols=("block",))
    cstats = scorer.candidate_stats(cands, gold, val_ids)
    print("[block]", json.dumps({**bstats, **cstats}, indent=1), f"({time.time()-t0:.0f}s)")

    scored = matcher.score_pairs(matcher.attach(cands, s1, others), args.w_name, args.missing_addr)
    lab = scored.merge(gold.assign(label=1), on=[C.S1, C.CAND], how="left")
    lab["label"] = lab["label"].fillna(0).astype(np.int8)
    lab.drop(columns=[c for c in lab if c.startswith(("name_n", "addr_n"))]).to_parquet(out / "scored_val.parquet")

    # coarse diagnostic sweep only; formal threshold tuning is B's final rung
    sweep = pd.DataFrame([{"threshold": t, **scorer.score(matcher.predict(scored, t), gold, val_ids)}
                          for t in np.round(np.arange(0.50, 1.001, 0.05), 2)])
    sweep.to_csv(out / "sweep.tsv", sep="\t", index=False)
    print(sweep[["threshold", "f_beta", "f_beta_singletons", "f_beta_nonsingletons",
                 "pair_precision", "pair_recall"]].to_string(index=False))

    pred = matcher.predict(scored, args.threshold)
    assert pred.merge(cands, on=[C.S1, C.CAND]).shape[0] == len(pred), "prediction outside candidate set"
    io.write_pairs(pred, out / "matching_results_val.tsv")
    res = scorer.score(pred, gold, val_ids)
    print("[final]", json.dumps(res, indent=1), f"({time.time()-t0:.0f}s)")

    best = sweep.loc[sweep["f_beta"].idxmax()]
    explog.append({
        "rung": "0", "owner": args.owner,
        "blocking": f"{blocker.NAME}(p={args.prefix_len},cap={args.max_block})",
        "matcher": f"{matcher.NAME}(w_name={args.w_name},miss={args.missing_addr})",
        "threshold": args.threshold,
        "candidate_recall": cstats["candidate_recall"], "cands_per_s1": cstats["cands_per_s1_mean"],
        "oracle_f05": cstats["oracle_f_beta"],
        "pair_precision": res["pair_precision"], "pair_recall": res["pair_recall"],
        "final_f05": res["f_beta"], "f05_singletons": res["f_beta_singletons"],
        "f05_nonsingletons": res["f_beta_nonsingletons"],
        "notes": f"coarse-sweep best t={best.threshold:.2f} F={best.f_beta:.4f}. {args.notes}".strip(),
    })


if __name__ == "__main__":
    main()
