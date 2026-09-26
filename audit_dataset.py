from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent / "student_resource"
DATASET = ROOT / "dataset"


FILES = {
    "train_source1": DATASET / "train" / "train_source1.tsv",
    "train_source2": DATASET / "train" / "train_source2.tsv",
    "train_source3": DATASET / "train" / "train_source3.tsv",
    "ground_truth": DATASET / "train" / "train_ground_truth.tsv",
    "test_source1": DATASET / "test" / "test_source1.tsv",
    "test_source2": DATASET / "test" / "test_source2.tsv",
    "test_source3": DATASET / "test" / "test_source3.tsv",
}


def inspect_file(name, path):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"path: {path}")

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        return

    print(f"size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    df = pd.read_csv(path, sep="\t")

    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")
    print(f"columns: {list(df.columns)}")

    print("\nmissing values:")
    print(df.isna().sum().to_string())

    print("\nunique values:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique(dropna=True):,}")

    print("\ndtypes:")
    print(df.dtypes.to_string())

    print("\nfirst 3 rows:")
    print(df.head(3).to_string(index=False))


def inspect_ground_truth(path):
    print("\n" + "=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(path, sep="\t", keep_default_na=False)

    print(f"rows: {len(df):,}")
    print(f"columns: {list(df.columns)}")

    matches = df["matched_entity_ids"].astype(str)

    match_counts = matches.apply(
        lambda x: 0 if not x.strip() else len(x.split(","))
    )

    print("\nmatches per Source 1:")
    print(match_counts.describe().to_string())

    print(f"\nsingletons: {(match_counts == 0).sum():,}")
    print(
        f"singleton percentage: "
        f"{(match_counts == 0).mean() * 100:.2f}%"
    )

    print(f"\n1 match: {(match_counts == 1).sum():,}")
    print(f"2 matches: {(match_counts == 2).sum():,}")
    print(f"3+ matches: {(match_counts >= 3).sum():,}")

    print("\nmatch-count distribution:")
    print(match_counts.value_counts().sort_index().to_string())


def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 - DATASET AUDIT")
    print("=" * 70)

    for name, path in FILES.items():
        if name != "ground_truth":
            inspect_file(name, path)

    inspect_ground_truth(FILES["ground_truth"])

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()