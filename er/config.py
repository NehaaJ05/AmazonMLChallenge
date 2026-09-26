"""Single source of truth for paths and column names. Edit HERE only.

Data dir resolution (first hit wins):
  1. env var ER_DATASET_DIR  (e.g. set ER_DATASET_DIR=C:\\...\\student_resource\\dataset)
  2. nearest parent directory containing dataset/train/train_source1.tsv
     (works whether this code sits in student_resource/ or student_resource/code/business_entity_resolution/src/)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _find_dataset() -> Path:
    if os.environ.get("ER_DATASET_DIR"):
        return Path(os.environ["ER_DATASET_DIR"]).resolve()
    for p in [ROOT, *ROOT.parents]:
        if (p / "dataset" / "train" / "train_source1.tsv").exists():
            return p / "dataset"
    return ROOT / "dataset"  # fallback; load will fail loudly with this path in the error


DATA = _find_dataset()
STUDENT_RESOURCE = DATA.parent
OUT = ROOT / "outputs"                        # experiment artifacts (val runs, caches)
SUBMISSION_DIR = STUDENT_RESOURCE / "output"  # final matching_results.tsv / candidate_pairs.tsv

TRAIN = {
    "S1": DATA / "train" / "train_source1.tsv",
    "S2": DATA / "train" / "train_source2.tsv",
    "S3": DATA / "train" / "train_source3.tsv",
}
TEST = {
    "S1": DATA / "test" / "test_source1.tsv",
    "S2": DATA / "test" / "test_source2.tsv",
    "S3": DATA / "test" / "test_source3.tsv",
}
GROUND_TRUTH = DATA / "train" / "train_ground_truth.tsv"

# ---- raw record columns (README) ----
ID, NAME, ADDR, COUNTRY = "entity_id", "business_name", "business_address", "country"

# ---- ground truth (README): one row per S1, comma-separated match list, empty = singleton ----
GT_S1_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"
GT_LIST_SEP = ","

# ---- official submission schema (README) ----
SUB_S1 = "source1_entity_id"
SUB_MATCH = "matched_entity_ids"      # matching_results.tsv
SUB_CAND = "candidate_entity_ids"     # candidate_pairs.tsv
SUB_LIST_SEP = ","
VALIDATOR = STUDENT_RESOURCE / "utils" / "validate_submission.py"

# ---- internal long-format pair schema (one row per S1/candidate pair) ----
S1 = "s1_id"
CAND = "cand_id"

# ---- frozen validation split (hash-based, by S1 entity) ----
VAL_FRACTION = 0.10
SPLIT_SALT = "amlc2026-val-v1"          # never change after freezing
VAL_IDS_FILE = ROOT / "splits" / "val_s1_ids.txt"

EXP_LOG = ROOT / "experiments" / "log.tsv"
BETA = 0.5
