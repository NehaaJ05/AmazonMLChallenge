"""Shared normalization. Rung 0 = deliberately BASIC (case, accents, punctuation, whitespace, '&').
Suffix expansion / address abbreviations / component extraction are Person A's rung 1, added here
so B's features pick them up automatically. Keep raw columns; only ADD *_n columns."""
import re
import unicodedata
import pandas as pd
import config as C

_COMBINING = re.compile(r"[\u0300-\u036f]")   # Latin diacritics only; keeps Devanagari matras intact
_NONWORD = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")


def _fold(s: str) -> str:
    return _COMBINING.sub("", unicodedata.normalize("NFKD", s))


def norm_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).map(_fold).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_NONWORD, " ", regex=True)
    return s.str.replace(_WS, " ", regex=True).str.strip()


def norm_country(s: pd.Series) -> pd.Series:
    # open-set: no whitelist, no mapping of known countries
    return s.fillna("").astype(str).str.strip().str.lower()


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_n"] = norm_text(df[C.NAME])
    df["addr_n"] = norm_text(df[C.ADDR])
    df["country_n"] = norm_country(df[C.COUNTRY])
    df["name_key"] = df["name_n"].str.replace(" ", "", regex=False)
    df["addr_missing"] = df["addr_n"].eq("")
    return df
