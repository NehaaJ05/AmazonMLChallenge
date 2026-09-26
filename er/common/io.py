import warnings
from pathlib import Path
import pandas as pd
import config as C


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path).astype(object)
    sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
    # Only a truly empty cell is missing; don't let pandas turn "NA"/"null" business text into NaN.
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def load_sources(files: dict) -> pd.DataFrame:
    frames = []
    for src, p in files.items():
        df = read_table(p)[[C.ID, C.NAME, C.ADDR, C.COUNTRY]].copy()
        df["source"] = src
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_dup = int(df[C.ID].duplicated().sum())
    if n_dup:
        warnings.warn(f"{n_dup} entity_ids repeat across sources; pairs are keyed on entity_id alone.")
    return df


def load_ground_truth(path=None) -> pd.DataFrame:
    gt = read_table(path or C.GROUND_TRUTH)[[C.GT_S1_COL, C.GT_MATCH_COL]]
    if C.GT_LIST_SEP:
        gt = gt.assign(**{C.GT_MATCH_COL: gt[C.GT_MATCH_COL].str.split(C.GT_LIST_SEP)}).explode(C.GT_MATCH_COL)
        gt[C.GT_MATCH_COL] = gt[C.GT_MATCH_COL].str.strip()
    gt = gt.rename(columns={C.GT_S1_COL: C.S1, C.GT_MATCH_COL: C.CAND})
    gt = gt.dropna(subset=[C.CAND])
    return gt[gt[C.CAND].ne("")].drop_duplicates().reset_index(drop=True)


def write_pairs(df: pd.DataFrame, path, extra_cols=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[[C.S1, C.CAND, *extra_cols]].to_csv(path, sep="\t", index=False)
