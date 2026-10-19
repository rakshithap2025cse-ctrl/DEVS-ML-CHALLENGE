"""Create the cleaned dataset: python clean_data.py --train train.csv --out train_cleaned.csv

Cleaning steps (all justified in eda.ipynb):
1. Parse `date` (DD/MM/YYYY HH:MM) and fix the 24:00 reading (NSM == 0) that is stamped with the wrong day
   -> new column `timestamp_corrected` (ISO format, strictly increasing, exact 15-minute spacing).
2. Verify: no missing values, no duplicates, target == next row's Usage_kWh.
3. Original columns are kept unchanged (no values dropped or clipped; high peaks are genuine production).
"""
import argparse
import pandas as pd
from features import true_timestamp, TARGET

ap = argparse.ArgumentParser(); ap.add_argument("--train", default="train.csv"); ap.add_argument("--out", default="train_cleaned.csv")
a = ap.parse_args()
df = pd.read_csv(a.train)
assert df.isna().sum().sum() == 0 and not df.duplicated().any()
ts = true_timestamp(df["date"], df["NSM"])
assert ts.is_monotonic_increasing and (ts.diff().dropna() == pd.Timedelta(minutes=15)).all()
assert (df[TARGET].iloc[:-1].values == df["Usage_kWh"].iloc[1:].values).all()
df.insert(1, "timestamp_corrected", ts.dt.strftime("%Y-%m-%d %H:%M:%S"))
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {df.shape[0]} rows, {df.shape[1]} columns; checks passed")
