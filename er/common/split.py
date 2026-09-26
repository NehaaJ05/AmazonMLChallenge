"""Frozen validation split over S1 entities. Hash-based => reproducible, but the written file is canonical."""
import hashlib
import pandas as pd
import config as C


def _bucket(x: str) -> float:
    return int(hashlib.md5(f"{C.SPLIT_SALT}:{x}".encode()).hexdigest()[:8], 16) / 2**32


def make_split(s1_ids: pd.Series) -> pd.Index:
    if C.VAL_IDS_FILE.exists():
        raise FileExistsError(f"{C.VAL_IDS_FILE} is frozen; delete deliberately (and tell your partner) to regenerate.")
    ids = pd.Series(pd.unique(s1_ids.astype(str)))
    val = ids[ids.map(_bucket) < C.VAL_FRACTION].sort_values()
    C.VAL_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.VAL_IDS_FILE.write_text("\n".join(val) + "\n")
    return pd.Index(val)


def load_val_ids() -> pd.Index:
    return pd.Index([l for l in C.VAL_IDS_FILE.read_text().splitlines() if l])


def get_or_make_val_ids(s1_ids: pd.Series) -> pd.Index:
    return load_val_ids() if C.VAL_IDS_FILE.exists() else make_split(s1_ids)
