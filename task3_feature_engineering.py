"""
Task 3: Feature Engineering
Create feature matrix by combining TF-IDF features, one-hot encoded sectors, and continuous features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import pandas as pd
import numpy as np
import pickle

print("=" * 60)
print("TASK 3: FEATURE ENGINEERING")
print("=" * 60)

# Load the labeled dataset
df = pd.read_csv('stock_news_labeled.csv')
print(f"\nDataset loaded: {len(df)} rows")

# 1. TF-IDF vectorization of article titles
print("\nCreating TF-IDF features from article titles...")
tfidf = TfidfVectorizer(
    max_features=100,  # Limit to top 100 words
    stop_words='english',
    min_df=2,  # Word must appear in at least 2 documents
    ngram_range=(1, 2)  # Use unigrams and bigrams
)

title_features = tfidf.fit_transform(df['Article_title'])
print(f"TF-IDF features shape: {title_features.shape}")

# Save the vectorizer for later use
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Saved TF-IDF vectorizer")

# 2. One-hot encode sectors
print("\nOne-hot encoding sectors...")
sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
print(f"Sector features shape: {sector_dummies.shape}")
print(f"Unique sectors: {sector_dummies.columns.tolist()}")

# 3. Normalize continuous features
print("\nNormalizing continuous features (market_cap, beta)...")
scaler = StandardScaler()
continuous_features = scaler.fit_transform(df[['market_cap', 'beta']])

# Save the scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"Continuous features shape: {continuous_features.shape}")
print("✓ Saved feature scaler")

# 4. Combine all features into one matrix
print("\nCombining all features...")
X = hstack([
    title_features,  # Sparse matrix (100 columns)
    sector_dummies.values,  # Dense array (~12 columns)
    continuous_features  # Dense array (2 columns)
])

y = df['label'].values

print(f"\n✓ Final feature matrix shape: {X.shape}")
print(f"  - TF-IDF features: 100")
print(f"  - Sector features: {sector_dummies.shape[1]}")
print(f"  - Continuous features: 2")
print(f"  - Total features: {X.shape[1]}")
print(f"\nTarget variable shape: {y.shape}")

# 5. Train-test split
print("\nPerforming train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Check label distribution in splits
print("\nTrain set label distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Label {label:2d}: {count:5d} ({count/len(y_train)*100:.1f}%)")

print("\nTest set label distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Label {label:2d}: {count:5d} ({count/len(y_test)*100:.1f}%)")

# Save train/test split for consistency across models
print("\nSaving train-test split...")
np.savez('train_test_split.npz', 
         X_train=X_train, X_test=X_test, 
         y_train=y_train, y_test=y_test)
print("✓ Saved train_test_split.npz")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 60)
