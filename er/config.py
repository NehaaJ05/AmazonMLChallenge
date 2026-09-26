"""Single source of truth for paths and column names. Edit HERE only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "student_resource" / "dataset"
OUT = ROOT / "outputs"

# ---- raw files: EDIT to match the actual dataset layout (csv / tsv / parquet all fine) ----
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

ID = "entity_id"
NAME = "business_name"
ADDR = "business_address"
COUNTRY = "country"

GT_S1_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"
GT_LIST_SEP = ","

S1 = "source1_entity_id"
CAND = "candidate_entity_ids"

# ---- frozen validation split (hash-based, by S1 entity) ----
VAL_FRACTION = 0.10
SPLIT_SALT = "amlc2026-val-v1"          # never change after freezing
VAL_IDS_FILE = ROOT / "splits" / "val_s1_ids.txt"

EXP_LOG = ROOT / "experiments" / "log.tsv"
BETA = 0.5
