"""Rung 0 on the TEST set -> student_resource/output/{matching_results,candidate_pairs}.tsv, then validate.
  python run_test.py --threshold 0.85        (use the best threshold from the val sweep)
S1 is processed in chunks so candidate pairs never all sit in memory at once."""
import argparse, subprocess, sys, time
import numpy as np
import pandas as pd
import config as C
from common import submission
from blocking import rung0 as blocker
from matcher import rung0 as matcher
from run_rung0 import load_normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--max-block", type=int, default=500)
    ap.add_argument("--w-name", type=float, default=0.6)
    ap.add_argument("--missing-addr", default="renorm")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()
    t0 = time.time()

    df = load_normalized(C.TEST, C.OUT / "cache" / "test_norm_rung0.parquet")
    s1_all, others = df[df["source"].eq("S1")], df[df["source"].ne("S1")]
    print(f"[load] test S1={len(s1_all):,} S2+S3={len(others):,} ({time.time()-t0:.0f}s)")

    cand_parts, pred_parts = [], []
    for i in range(0, len(s1_all), args.chunk):
        s1 = s1_all.iloc[i:i + args.chunk]
        cands, _ = blocker.generate(s1, others, args.prefix_len, args.max_block)
        scored = matcher.score_pairs(matcher.attach(cands, s1, others), args.w_name, args.missing_addr)
        cand_parts.append(cands[[C.S1, C.CAND]])
        pred_parts.append(matcher.predict(scored, args.threshold))
        print(f"  chunk {i//args.chunk}: {len(cands):,} cands, {len(pred_parts[-1]):,} matches ({time.time()-t0:.0f}s)")
    cands, pred = pd.concat(cand_parts), pd.concat(pred_parts)

    s1_ids = s1_all[C.ID].tolist()
    mpath, cpath = C.SUBMISSION_DIR / "matching_results.tsv", C.SUBMISSION_DIR / "candidate_pairs.tsv"
    submission.write(pred, s1_ids, mpath, C.SUB_MATCH)
    submission.write(cands, s1_ids, cpath, C.SUB_CAND)
    per = pred.groupby(C.S1).size().reindex(s1_ids, fill_value=0)
    print(f"[write] {mpath}\n  cands/S1={len(cands)/len(s1_ids):.1f}  matches/S1={per.mean():.2f}  "
          f"empty-prediction rate={per.eq(0).mean():.3f} (train singleton rate is ~0.056)")
    print("  by country:", pred.merge(s1_all[[C.ID, 'country_n']], left_on=C.S1, right_on=C.ID)
          .groupby("country_n").size().to_dict())

    issues = submission.check(mpath, cpath, s1_ids, others[C.ID])
    print("[local check]", "PASS" if not issues else "\n  " + "\n  ".join(issues))
    if C.VALIDATOR.exists():
        r = subprocess.run([sys.executable, str(C.VALIDATOR), "--matching", str(mpath), "--candidate", str(cpath),
                            "--test-dir", str(C.DATA / "test")], cwd=C.STUDENT_RESOURCE)
        print(f"[official validator] exit code {r.returncode}")
    else:
        print(f"[official validator] not found at {C.VALIDATOR}")


if __name__ == "__main__":
    main()
