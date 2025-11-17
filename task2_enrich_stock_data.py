"""
Task 2: Enrich with Stock Data from yfinance
Download stock metadata and price data, calculate next-day returns, and label examples.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 60)
print("TASK 2: ENRICH WITH STOCK DATA FROM YFINANCE")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv('fnspid_2020_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"\nDataset loaded: {len(df)} rows")

# Get unique stocks
unique_stocks = df['Stock_symbol'].unique()
print(f"Number of unique stocks: {len(unique_stocks)}")

# Download stock data for all of 2020 + early 2021 (for next-day prices)
print("\nDownloading stock price data and metadata...")
print("(This will take several minutes...)")

stock_data = {}
stock_metadata = {}
failed_stocks = []

for i, symbol in enumerate(unique_stocks):
    if i % 100 == 0:
        print(f"  Processing stock {i}/{len(unique_stocks)}: {symbol}")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Download price data
        hist = ticker.history(start='2020-01-01', end='2021-01-31')
        if len(hist) > 0:
            stock_data[symbol] = hist
            
            # Get metadata
            info = ticker.info
            stock_metadata[symbol] = {
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', np.nan),
                'beta': info.get('beta', np.nan)
            }
        else:
            failed_stocks.append(symbol)
            
    except Exception as e:
        failed_stocks.append(symbol)
        continue

print(f"\n✓ Successfully downloaded {len(stock_data)} stocks")
print(f"✗ Failed stocks: {len(failed_stocks)}")

# Calculate next-day returns and labels
print("\nCalculating next-day returns...")

def get_next_day_return(symbol, date, stock_data):
    """
    Get the next trading day's return for a given stock and date.
    Returns None if data not available.
    """
    if symbol not in stock_data:
        return None
    
    hist = stock_data[symbol]
    
    # Find the date in the historical data
    try:
        # Get the close price on the article date
        if date not in hist.index:
            # Find the next available trading day
            future_dates = hist.index[hist.index >= date]
            if len(future_dates) == 0:
                return None
            date = future_dates[0]
        
        current_price = hist.loc[date, 'Close']
        
        # Get next trading day
        future_dates = hist.index[hist.index > date]
        if len(future_dates) == 0:
            return None
        
        next_date = future_dates[0]
        next_price = hist.loc[next_date, 'Close']
        
        # Calculate return
        return_pct = (next_price - current_price) / current_price
        return return_pct
        
    except Exception as e:
        return None

# Calculate returns for all rows
df['next_day_return'] = df.apply(
    lambda row: get_next_day_return(row['Stock_symbol'], row['Date'], stock_data),
    axis=1
)

# Remove rows where we couldn't get returns
before_count = len(df)
df = df.dropna(subset=['next_day_return'])
print(f"Rows with valid returns: {len(df)} (removed {before_count - len(df)})")

# Add stock metadata to each row
print("\nAdding stock metadata...")

def get_stock_metadata(symbol, stock_metadata):
    if symbol in stock_metadata:
        return stock_metadata[symbol]
    return {'sector': 'Unknown', 'market_cap': np.nan, 'beta': np.nan}

df['sector'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['sector'])
df['market_cap'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['market_cap'])
df['beta'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['beta'])

# Remove rows with missing metadata
before_count = len(df)
df = df.dropna(subset=['sector', 'market_cap', 'beta'])
print(f"Rows after adding metadata: {len(df)} (removed {before_count - len(df)})")

# Create labels using ±1.5% threshold
print("\nCreating labels...")

def label_return(return_pct):
    if return_pct > 0.015:
        return 1  # Positive
    elif return_pct < -0.015:
        return -1  # Negative
    else:
        return 0  # Neutral

df['label'] = df['next_day_return'].apply(label_return)

# Check label distribution
print("\nLabel distribution:")
print(df['label'].value_counts().sort_index())
print("\nLabel distribution (%):")
print(df['label'].value_counts(normalize=True).sort_index() * 100)

# Save enriched dataset
df.to_csv('stock_news_labeled.csv', index=False)
print(f"\n✓ Saved labeled dataset with {len(df)} examples")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Unique sectors: {df['sector'].nunique()}")
