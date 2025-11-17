#!/usr/bin/env python3

from datasets import load_dataset
import pandas as pd

def main():
    # 1. Load the dataset from Hugging Face
    #    (adjust split if needed, but FNSPID uses "train")
    print("Loading FNSPID dataset from Hugging Face...")
    ds = load_dataset("Zihan1004/FNSPID", split="train")

    # 2. Convert to pandas DataFrame
    print("Converting to pandas DataFrame...")
    df = ds.to_pandas()

    # 3. Make sure the date column exists
    #    Common name is "Date" – adjust if your column name differs
    date_col = "Date"
    if date_col not in df.columns:
        raise ValueError(f"Could not find date column '{date_col}'. "
                         f"Available columns: {list(df.columns)}")

    # 4. Parse dates & filter to year 2020
    print("Filtering to events in 2020...")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df_2020 = df[df[date_col].dt.year == 2020].copy()

    if len(df_2020) == 0:
        raise ValueError("No rows found for year 2020. "
                         "Check the date format or column name.")

    # 5. Take 100 events (or all if fewer than 100)
    n_samples = min(100, len(df_2020))
    print(f"Sampling {n_samples} events from {len(df_2020)} total 2020 events...")
    df_2020_100 = df_2020.sample(n=n_samples, random_state=42)

    # 6. Save to CSV
    out_path = "fnspid_2020_100.csv"
    df_2020_100.to_csv(out_path, index=False)
    print(f"Saved {n_samples} events to {out_path}")

if __name__ == "__main__":
    main()
