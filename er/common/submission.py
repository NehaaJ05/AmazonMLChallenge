"""Long pairs <-> official submission files (README 'Output Format').
One row per S1 (empties included), comma-joined S2-/S3- IDs, no duplicates, tab-separated, '\n' endings."""
from pathlib import Path
import pandas as pd
import config as C


def to_lists(pairs: pd.DataFrame, s1_ids) -> pd.Series:
    """Series indexed by every S1 id (input order kept) -> sorted, de-duplicated comma-joined string ('' if none)."""
    ids = pd.Index(pd.unique(pd.Series(list(s1_ids), dtype=object)))
    p = pairs[[C.S1, C.CAND]].drop_duplicates()
    p = p[p[C.S1].isin(ids)].sort_values([C.S1, C.CAND])
    joined = p.groupby(C.S1)[C.CAND].agg(C.SUB_LIST_SEP.join)
    return joined.reindex(ids, fill_value="")


def write(pairs: pd.DataFrame, s1_ids, path, list_col: str):
    """Write matching_results.tsv (list_col=C.SUB_MATCH) or candidate_pairs.tsv (list_col=C.SUB_CAND)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lists = to_lists(pairs, s1_ids)
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # explicit: no \r\n on Windows, no quoting
        f.write(f"{C.SUB_S1}\t{list_col}\n")
        for s1, ids in lists.items():
            f.write(f"{s1}\t{ids}\n")


def read(path, list_col: str) -> pd.DataFrame:
    """Inverse of write(): back to long pairs. Also used for the ground truth, which has the same shape."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.assign(**{list_col: df[list_col].str.split(C.SUB_LIST_SEP)}).explode(list_col)
    df = df[df[list_col].fillna("").str.strip().ne("")]
    return df.rename(columns={C.SUB_S1: C.S1, list_col: C.CAND})[[C.S1, C.CAND]].reset_index(drop=True)


def check(match_path, cand_path, s1_ids, valid_other_ids) -> list:
    """Local mirror of the README rules; the official utils/validate_submission.py is still the authority."""
    issues, s1_set, other_set = [], set(s1_ids), set(valid_other_ids)
    for path, col in [(match_path, C.SUB_MATCH), (cand_path, C.SUB_CAND)]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if list(df.columns) != [C.SUB_S1, col]:
            issues.append(f"{path.name}: header {list(df.columns)}")
        if df[C.SUB_S1].duplicated().any():
            issues.append(f"{path.name}: duplicate S1 rows")
        if set(df[C.SUB_S1]) != s1_set:
            issues.append(f"{path.name}: S1 set mismatch (missing {len(s1_set - set(df[C.SUB_S1]))})")
        lists = df[col].map(lambda s: [x for x in s.split(C.SUB_LIST_SEP) if x])
        if (lists.map(len) != lists.map(lambda l: len(set(l)))).any():
            issues.append(f"{path.name}: duplicate IDs within a list")
        bad = {x for l in lists for x in l} - other_set
        if bad:
            issues.append(f"{path.name}: {len(bad)} IDs not in test S2/S3, e.g. {sorted(bad)[:3]}")
    m, c = read(match_path, C.SUB_MATCH), read(cand_path, C.SUB_CAND)
    if len(m.merge(c, on=[C.S1, C.CAND])) != len(m):
        issues.append("matches not a subset of candidates")
    return issues
