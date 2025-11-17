"""
Task 1: Load and Filter FNSPID Dataset
Load 2020 news data from HuggingFace and perform initial filtering.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

print("=" * 60)
print("TASK 1: LOAD AND FILTER FNSPID DATASET")
print("=" * 60)

# Download the CSV file from HuggingFace
print("\nDownloading FNSPID dataset from HuggingFace...")
file_path = hf_hub_download(
    repo_id="Zihan1004/FNSPID",
    filename="Stock_news/All_external.csv",
    repo_type="dataset"
)

# Load with pandas, being flexible with data types
print("Loading CSV with pandas...")
df = pd.read_csv(file_path, low_memory=False)
print(f"Total rows loaded: {len(df)}")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter for year 2020
df_2020 = df[df['Date'].dt.year == 2020].copy()
print(f"Total rows in 2020: {len(df_2020)}")

# Keep only necessary columns
df_2020 = df_2020[['Date', 'Article_title', 'Stock_symbol']].copy()

# Remove rows with null article titles
df_2020 = df_2020.dropna(subset=['Article_title'])
print(f"Rows after removing null titles: {len(df_2020)}")

# Remove rows with null stock symbols
df_2020 = df_2020.dropna(subset=['Stock_symbol'])
print(f"Rows after removing null symbols: {len(df_2020)}")

# Sample if dataset is too large (optional - if >20,000 rows)
if len(df_2020) > 20000:
    df_2020 = df_2020.sample(n=20000, random_state=42)
    print(f"Sampled to {len(df_2020)} rows")

# Save cleaned dataset
df_2020.to_csv('fnspid_2020_cleaned.csv', index=False)
print(f"\n✓ Saved cleaned dataset to fnspid_2020_cleaned.csv")
print(f"  Final row count: {len(df_2020)}")
print(f"  Unique stocks: {df_2020['Stock_symbol'].nunique()}")
print(f"  Date range: {df_2020['Date'].min()} to {df_2020['Date'].max()}")
