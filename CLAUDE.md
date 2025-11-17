# Event-Driven Stock Impact Classifier - Implementation Guide

## Project Overview
This project predicts how individual stocks react to news events using multi-class classification. We classify stock reactions as Positive (+1), Neutral (0), or Negative (-1) based on next-day price movements.

**Input**: News article data (date, title, stock symbol) + Stock characteristics (sector, market cap, beta)  
**Output**: Three-way classification: +1 (price increase >1.5%), -1 (price decrease <-1.5%), 0 (otherwise)  
**Models**: Logistic Regression (baseline), Random Forest, XGBoost

---

## Task Breakdown

### Task 1: Load and Filter FNSPID Dataset (1 hour)

**Objective**: Load 2020 news data from HuggingFace and perform initial filtering.

**Steps**:
1. Install required libraries:
```bash
pip install datasets pandas numpy scikit-learn yfinance xgboost matplotlib seaborn --break-system-packages
```

2. Load the FNSPID dataset from HuggingFace:
```python
from datasets import load_dataset
import pandas as pd

# Load only the all_external.csv file
dataset = load_dataset("Zihan1004/FNSPID", data_files="all_external.csv", split="train")

# Convert to pandas DataFrame
df = dataset.to_pandas()
```

3. Filter for 2020 data:
```python
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter for year 2020
df_2020 = df[df['Date'].dt.year == 2020].copy()

print(f"Total rows in 2020: {len(df_2020)}")
```

4. Clean the data:
```python
# Keep only necessary columns
df_2020 = df_2020[['Date', 'Article_title', 'Stock_symbol']].copy()

# Remove rows with null article titles
df_2020 = df_2020.dropna(subset=['Article_title'])

# Remove rows with null stock symbols
df_2020 = df_2020.dropna(subset=['Stock_symbol'])

print(f"Rows after cleaning: {len(df_2020)}")
```

5. Sample if dataset is too large (optional - if >20,000 rows):
```python
# If we have more than 20,000 rows, sample for faster processing
if len(df_2020) > 20000:
    df_2020 = df_2020.sample(n=20000, random_state=42)
    print(f"Sampled to {len(df_2020)} rows")
```

6. Save cleaned dataset:
```python
df_2020.to_csv('fnspid_2020_cleaned.csv', index=False)
print("Saved cleaned dataset to fnspid_2020_cleaned.csv")
```

**Expected Output**: CSV file with ~10,000-20,000 rows containing Date, Article_title, Stock_symbol

---

### Task 2: Enrich with Stock Data from yfinance (2-3 hours)

**Objective**: Download stock metadata and price data, calculate next-day returns, and label examples.

**Important Notes**:
- Some stocks may not be available on yfinance (delisted, wrong symbols, etc.)
- Weekend/holiday dates won't have trading data - we need to handle this
- This task will take time due to API calls - be patient

**Steps**:

1. Get unique stock symbols and download their data:
```python
import yfinance as yf
import numpy as np
from datetime import timedelta

# Load cleaned dataset
df = pd.read_csv('fnspid_2020_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get unique stocks
unique_stocks = df['Stock_symbol'].unique()
print(f"Number of unique stocks: {len(unique_stocks)}")

# Download stock data for all of 2020 + early 2021 (for next-day prices)
stock_data = {}
stock_metadata = {}
failed_stocks = []

for i, symbol in enumerate(unique_stocks):
    if i % 50 == 0:
        print(f"Processing stock {i}/{len(unique_stocks)}: {symbol}")
    
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

print(f"Successfully downloaded {len(stock_data)} stocks")
print(f"Failed stocks: {len(failed_stocks)}")
```

2. Calculate next-day returns and labels:
```python
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
df = df.dropna(subset=['next_day_return'])
print(f"Rows with valid returns: {len(df)}")
```

3. Add stock metadata to each row:
```python
def get_stock_metadata(symbol, stock_metadata):
    if symbol in stock_metadata:
        return stock_metadata[symbol]
    return {'sector': 'Unknown', 'market_cap': np.nan, 'beta': np.nan}

df['sector'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['sector'])
df['market_cap'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['market_cap'])
df['beta'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['beta'])

# Remove rows with missing metadata
df = df.dropna(subset=['sector', 'market_cap', 'beta'])
print(f"Rows after adding metadata: {len(df)}")
```

4. Create labels using ±1.5% threshold:
```python
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
print(df['label'].value_counts())
print("\nLabel distribution (%):")
print(df['label'].value_counts(normalize=True) * 100)
```

5. Save enriched dataset:
```python
df.to_csv('stock_news_labeled.csv', index=False)
print(f"\nSaved labeled dataset with {len(df)} examples")
print(f"Columns: {df.columns.tolist()}")
```

**Expected Output**: 
- CSV with columns: Date, Article_title, Stock_symbol, next_day_return, sector, market_cap, beta, label
- Should have 5,000-15,000 rows (some will be filtered out due to missing data)
- Label distribution should be somewhat balanced (ideally 30-40% each class)

---

### Task 3: Feature Engineering (1 hour)

**Objective**: Create feature matrix by combining TF-IDF features, one-hot encoded sectors, and continuous features.

**Steps**:

1. Load the labeled dataset:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd

df = pd.read_csv('stock_news_labeled.csv')
print(f"Dataset loaded: {len(df)} rows")
```

2. TF-IDF vectorization of article titles:
```python
# Create TF-IDF features from article titles
tfidf = TfidfVectorizer(
    max_features=100,  # Limit to top 100 words
    stop_words='english',
    min_df=2,  # Word must appear in at least 2 documents
    ngram_range=(1, 2)  # Use unigrams and bigrams
)

title_features = tfidf.fit_transform(df['Article_title'])
print(f"TF-IDF features shape: {title_features.shape}")

# Save the vectorizer for later use
import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
```

3. One-hot encode sectors:
```python
# One-hot encode sector
sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
print(f"Sector features shape: {sector_dummies.shape}")
print(f"Unique sectors: {sector_dummies.columns.tolist()}")
```

4. Normalize continuous features:
```python
from sklearn.preprocessing import StandardScaler

# Normalize market_cap and beta
scaler = StandardScaler()
continuous_features = scaler.fit_transform(df[['market_cap', 'beta']])

# Save the scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Continuous features shape: {continuous_features.shape}")
```

5. Combine all features:
```python
# Combine all features into one matrix
# Note: hstack can handle both sparse and dense matrices
X = hstack([
    title_features,  # Sparse matrix (100 columns)
    sector_dummies.values,  # Dense array (~11 columns)
    continuous_features  # Dense array (2 columns)
])

y = df['label'].values

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Total features: {X.shape[1]}")
```

6. Train-test split:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Save train/test split for consistency across models
import numpy as np
np.savez('train_test_split.npz', 
         X_train=X_train, X_test=X_test, 
         y_train=y_train, y_test=y_test)
```

**Expected Output**:
- Feature matrix X with shape (n_samples, ~113 features)
- Saved train/test split in .npz file
- Saved TF-IDF vectorizer and scaler in .pkl files

---

### Task 4: Train and Evaluate Models (2-3 hours)

**Objective**: Train three models (Logistic Regression, Random Forest, XGBoost) and evaluate performance.

**Steps**:

1. Load the train/test split:
```python
import numpy as np
from scipy import sparse

# Load train/test split
data = np.load('train_test_split.npz', allow_pickle=True)

# Handle both dense and sparse matrices
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Convert to sparse if needed (for efficiency with TF-IDF features)
if not sparse.issparse(X_train):
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)

print(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
```

2. Train Logistic Regression (Baseline):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='multinomial',
    solver='lbfgs'
)
lr_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Negative (-1)', 'Neutral (0)', 'Positive (+1)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
```

3. Train Random Forest:
```python
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Negative (-1)', 'Neutral (0)', 'Positive (+1)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
```

4. Train XGBoost:
```python
import xgboost as xgb

print("\n" + "="*50)
print("XGBOOST")
print("="*50)

# XGBoost expects labels to be 0, 1, 2 (not -1, 0, 1)
# Map labels: -1 -> 0, 0 -> 1, 1 -> 2
y_train_xgb = y_train + 1
y_test_xgb = y_test + 1

start_time = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    objective='multi:softmax',
    num_class=3
)
xgb_model.fit(X_train, y_train_xgb)
train_time = time.time() - start_time

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Map predictions back: 0 -> -1, 1 -> 0, 2 -> 1
y_pred_xgb_original = y_pred_xgb - 1

# Evaluation
accuracy_xgb = accuracy_score(y_test, y_pred_xgb_original)
print(f"\nTraining time: {train_time:.2f} seconds")
print(f"Accuracy: {accuracy_xgb:.4f} ({accuracy_xgb*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb_original, target_names=['Negative (-1)', 'Neutral (0)', 'Positive (+1)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb_original))
```

5. Save model results:
```python
import pickle

# Save models
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
    
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
    
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save results summary
results = {
    'Logistic Regression': accuracy_lr,
    'Random Forest': accuracy_rf,
    'XGBoost': accuracy_xgb
}

with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nModels saved successfully!")
```

**Expected Output**:
- Three trained models with accuracy scores
- Classification reports and confusion matrices for each
- Saved model files (.pkl)

---

### Task 5: Create Visualizations (1 hour)

**Objective**: Generate visualizations for the milestone report.

**Steps**:

1. Create comparison bar chart:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load results
with open('model_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Create bar chart
plt.figure(figsize=(10, 6))
models = list(results.keys())
accuracies = [results[m] * 100 for m in models]

bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 100])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=11)

# Add baseline reference line
plt.axhline(y=33.33, color='red', linestyle='--', label='Random Baseline (33.33%)')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved model_comparison.png")
plt.close()
```

2. Create confusion matrices:
```python
from sklearn.metrics import confusion_matrix
import numpy as np

# Load data
data = np.load('train_test_split.npz', allow_pickle=True)
y_test = data['y_test']

# Load models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Get predictions
X_test = data['X_test']
if not sparse.issparse(X_test):
    from scipy import sparse
    X_test = sparse.csr_matrix(X_test)

y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test) - 1  # Convert back to -1, 0, 1

# Create confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_data = [
    ('Logistic Regression', y_pred_lr),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb)
]

for idx, (name, y_pred) in enumerate(models_data):
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved confusion_matrices.png")
plt.close()
```

3. Create label distribution visualization:
```python
import pandas as pd

df = pd.read_csv('stock_news_labeled.csv')

# Label distribution
label_counts = df['label'].value_counts().sort_index()
label_names = ['Negative (-1)', 'Neutral (0)', 'Positive (+1)']
label_values = [label_counts.get(-1, 0), label_counts.get(0, 0), label_counts.get(1, 0)]

plt.figure(figsize=(10, 6))
bars = plt.bar(label_names, label_values, color=['#d62728', '#7f7f7f', '#2ca02c'])
plt.ylabel('Number of Examples', fontsize=12)
plt.xlabel('Label', fontsize=12)
plt.title('Dataset Label Distribution', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, label_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val}\n({val/sum(label_values)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
print("Saved label_distribution.png")
plt.close()
```

4. Create dataset sample visualization (for proof of data):
```python
# Show a sample of the dataset
sample_df = df[['Date', 'Article_title', 'Stock_symbol', 'sector', 'next_day_return', 'label']].head(10)

# Create table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Format the data for display
display_data = sample_df.copy()
display_data['next_day_return'] = display_data['next_day_return'].apply(lambda x: f'{x*100:.2f}%')
display_data['Article_title'] = display_data['Article_title'].str[:50] + '...'  # Truncate titles

table = ax.table(cellText=display_data.values,
                colLabels=display_data.columns,
                cellLoc='left',
                loc='center',
                colWidths=[0.12, 0.40, 0.10, 0.15, 0.13, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style the header
for i in range(len(display_data.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Sample Dataset Rows (First 10)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('dataset_sample.png', dpi=300, bbox_inches='tight')
print("Saved dataset_sample.png")
plt.close()

print("\nAll visualizations created successfully!")
```

**Expected Output**:
- model_comparison.png: Bar chart comparing model accuracies
- confusion_matrices.png: Three confusion matrices side-by-side
- label_distribution.png: Bar chart of label distribution
- dataset_sample.png: Table showing sample data

---

## Milestone 2 Report Structure

Once all tasks are complete, create a PDF report with the following sections:

### 1. Brief Project Recap (2-3 sentences)
"Our project predicts how individual stocks react to news events using multi-class classification. We use financial news data paired with stock characteristics (sector, market cap, beta) to classify reactions as Positive (+1), Neutral (0), or Negative (-1) based on next-day price movements exceeding ±1.5%. We implemented three supervised learning models: Logistic Regression, Random Forest, and XGBoost."

### 2. Significant Accomplishments

**Accomplishment 1: Dataset Creation and Labeling**
- Loaded 2020 financial news data from FNSPID dataset (HuggingFace)
- Downloaded historical price data for stocks using yfinance API
- Generated X,XXX labeled training examples by calculating next-day returns
- Implemented systematic labeling with ±1.5% threshold
- **Proof**: Include `dataset_sample.png` and `label_distribution.png`

**Accomplishment 2: Feature Engineering**
- Implemented TF-IDF vectorization on news article titles (100 features)
- One-hot encoded stock sectors (~11 features)
- Normalized continuous features (market cap, beta)
- Created combined feature matrix with ~113 total features
- **Proof**: Include code snippet showing feature engineering and final matrix shape

**Accomplishment 3: Model Training and Evaluation**
- Trained three models: Logistic Regression (baseline), Random Forest, XGBoost
- Achieved X% accuracy with Logistic Regression (vs 33.33% random baseline)
- Random Forest achieved X% accuracy
- XGBoost achieved X% accuracy
- **Proof**: Include `model_comparison.png` and `confusion_matrices.png`

### 3. Challenges or Roadblocks
- **Data Availability**: Some stocks in FNSPID were not available on yfinance (delisted or incorrect symbols). Filtered out ~X% of rows.
- **Weekend/Holiday Dates**: News published on non-trading days required finding next available trading day for price data.
- **Label Imbalance**: Initial 2% threshold resulted in too many neutral classifications. Adjusted to 1.5% for better balance.
- **Processing Time**: Downloading price data for thousands of stocks took significant time. Implemented progress tracking and error handling.

### 4. Changes from Original Plan
- **What changed**: Used FNSPID's individual stock-news pairs directly instead of identifying "major market events." This resulted in a more diverse dataset with news for individual stocks rather than market-wide events.
- **Why**: This approach is more systematic, scalable, and better aligned with the dataset structure. Each (news, stock) pair is already a natural training example.
- **Impact**: Simplified data pipeline and increased dataset diversity. Models now learn from a broader range of news events rather than just major market catalysts.

---

## Expected Performance Targets

Based on the project proposal:
- **Target Accuracy**: 45-50% (significantly above 33% random baseline)
- **Acceptable Range**: 40-60%

If accuracy is below 40%, consider:
- Checking for data quality issues
- Verifying label distribution is balanced
- Ensuring features are properly normalized
- Trying different TF-IDF parameters (max_features, ngram_range)

---

## Files That Should Exist After Completion

1. **Data Files**:
   - `fnspid_2020_cleaned.csv` - Cleaned news data
   - `stock_news_labeled.csv` - Final labeled dataset
   - `train_test_split.npz` - Train/test split

2. **Model Files**:
   - `tfidf_vectorizer.pkl` - TF-IDF vectorizer
   - `feature_scaler.pkl` - Feature scaler
   - `logistic_regression_model.pkl` - Trained LR model
   - `random_forest_model.pkl` - Trained RF model
   - `xgboost_model.pkl` - Trained XGB model
   - `model_results.pkl` - Results summary

3. **Visualization Files**:
   - `model_comparison.png`
   - `confusion_matrices.png`
   - `label_distribution.png`
   - `dataset_sample.png`

4. **Code Files**:
   - Scripts for each task (recommended to organize as separate .py files)

---

## Troubleshooting

### If dataset is too small (<2000 examples):
- Lower the return threshold to ±1.0%
- Include more stocks (relax filtering criteria)
- Use data from 2019 as well

### If training is too slow:
- Reduce TF-IDF max_features to 50
- Reduce Random Forest n_estimators to 50
- Sample dataset to 10,000 rows

### If accuracy is very low (<35%):
- Check label distribution (should not be >70% one class)
- Verify next-day return calculations
- Check for data leakage or errors
- Try different feature combinations

---

## Summary Checklist

- [ ] Task 1: FNSPID data loaded and filtered for 2020
- [ ] Task 2: Stock data downloaded, returns calculated, labels created
- [ ] Task 3: Features engineered (TF-IDF + one-hot + continuous)
- [ ] Task 4: Three models trained and evaluated
- [ ] Task 5: All visualizations created
- [ ] All expected files exist
- [ ] Accuracy above 40% for at least one model
- [ ] Ready to write milestone report

Good luck! This is a solid project that demonstrates real machine learning skills.
