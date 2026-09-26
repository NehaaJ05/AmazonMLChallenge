"""Append-only shared experiment log. One row per (rung, current full pipeline)."""
import csv, datetime, subprocess
import config as C

COLS = ["timestamp", "git", "rung", "owner", "blocking", "matcher", "threshold",
        "candidate_recall", "cands_per_s1", "oracle_f05", "pair_precision", "pair_recall",
        "final_f05", "f05_singletons", "f05_nonsingletons", "notes"]


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def append(row: dict):
    C.EXP_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not C.EXP_LOG.exists()
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "git": _git(), **row}
    fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
    with open(C.EXP_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(fmt)
