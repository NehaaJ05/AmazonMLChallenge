"""Rung 0 blocking: (country, exact normalized name key) UNION (country, name-key prefix).
Oversized prefix blocks are skipped (exact blocks never are) and the loss is reported, not hidden."""
import pandas as pd
import config as C

NAME = "rung0_exact+prefix"


def generate(s1: pd.DataFrame, others: pd.DataFrame, prefix_len: int = 8,
             max_block: int = 500, min_key_len: int = 3):
    a = s1.loc[s1["name_key"].str.len() >= min_key_len, [C.ID, "country_n", "name_key"]]
    b = others.loc[others["name_key"].str.len() >= min_key_len, [C.ID, "country_n", "name_key"]]

    exact = a.merge(b, on=["country_n", "name_key"], suffixes=("_s1", "_c"))
    exact = exact[[f"{C.ID}_s1", f"{C.ID}_c"]].assign(block="exact")

    a = a.assign(pk=a["name_key"].str[:prefix_len])
    b = b.assign(pk=b["name_key"].str[:prefix_len])
    size = b.groupby(["country_n", "pk"])[C.ID].transform("size")
    big_keys = b.loc[size > max_block, ["country_n", "pk"]].drop_duplicates()
    s1_in_big = a.merge(big_keys, on=["country_n", "pk"])
    prefix = a.merge(b[size <= max_block], on=["country_n", "pk"], suffixes=("_s1", "_c"))
    prefix = prefix[[f"{C.ID}_s1", f"{C.ID}_c"]].assign(block="prefix")

    cands = (pd.concat([exact, prefix], ignore_index=True)
             .rename(columns={f"{C.ID}_s1": C.S1, f"{C.ID}_c": C.CAND})
             .drop_duplicates([C.S1, C.CAND], keep="first")   # exact wins the label
             .reset_index(drop=True))
    stats = {
        "n_oversized_prefix_blocks": int(len(big_keys)),
        "s1_rows_hitting_oversized_block": int(len(s1_in_big)),
        "pairs_exact": int(cands["block"].eq("exact").sum()),
        "pairs_prefix_only": int(cands["block"].eq("prefix").sum()),
    }
    return cands, stats
