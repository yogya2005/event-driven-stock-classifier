"""
Task 4: Train and Evaluate Models
Train three models (Logistic Regression, Random Forest, XGBoost) and evaluate performance.
"""

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import time
import pickle

print("=" * 70)
print("TASK 4: TRAIN AND EVALUATE MODELS")
print("=" * 70)

# Load the train/test split
print("\nLoading train/test split...")
data = np.load('train_test_split.npz', allow_pickle=True)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Data is already in sparse format from scipy.sparse.hstack()
# Just need to convert from object array back to sparse matrix
if isinstance(X_train, np.ndarray) and X_train.dtype == object:
    X_train = X_train.item()
    X_test = X_test.item()

print(f"Loaded data - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Baseline (random) accuracy: 33.33%")

# =============================================================================
# 1. LOGISTIC REGRESSION (BASELINE)
# =============================================================================
print("\n" + "=" * 70)
print("1. LOGISTIC REGRESSION")
print("=" * 70)

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
print("           Pred:-1  Pred:0  Pred:+1")
cm_lr = confusion_matrix(y_test, y_pred_lr)
labels = ['True:-1', 'True:0 ', 'True:+1']
for i, label in enumerate(labels):
    print(f"{label}  {cm_lr[i][0]:7d}  {cm_lr[i][1]:6d}  {cm_lr[i][2]:7d}")

# =============================================================================
# 2. RANDOM FOREST
# =============================================================================
print("\n" + "=" * 70)
print("2. RANDOM FOREST")
print("=" * 70)

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=0
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
print("           Pred:-1  Pred:0  Pred:+1")
cm_rf = confusion_matrix(y_test, y_pred_rf)
for i, label in enumerate(labels):
    print(f"{label}  {cm_rf[i][0]:7d}  {cm_rf[i][1]:6d}  {cm_rf[i][2]:7d}")

# =============================================================================
# 3. XGBOOST
# =============================================================================
print("\n" + "=" * 70)
print("3. XGBOOST")
print("=" * 70)

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
    num_class=3,
    verbosity=0
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
print("           Pred:-1  Pred:0  Pred:+1")
cm_xgb = confusion_matrix(y_test, y_pred_xgb_original)
for i, label in enumerate(labels):
    print(f"{label}  {cm_xgb[i][0]:7d}  {cm_xgb[i][1]:6d}  {cm_xgb[i][2]:7d}")

# =============================================================================
# SAVE MODELS AND RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS AND RESULTS")
print("=" * 70)

# Save models
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✓ Saved logistic_regression_model.pkl")
    
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ Saved random_forest_model.pkl")
    
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✓ Saved xgboost_model.pkl")

# Save results summary
results = {
    'Logistic Regression': accuracy_lr,
    'Random Forest': accuracy_rf,
    'XGBoost': accuracy_xgb
}

with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ Saved model_results.pkl")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(f"Baseline (random):      33.33%")
print(f"Logistic Regression:    {accuracy_lr*100:.2f}%")
print(f"Random Forest:          {accuracy_rf*100:.2f}%")
print(f"XGBoost:                {accuracy_xgb*100:.2f}%")
print("=" * 70)

# Determine best model
best_model = max(results, key=results.get)
best_accuracy = results[best_model]
print(f"\n🏆 Best model: {best_model} with {best_accuracy*100:.2f}% accuracy")
print(f"   Improvement over baseline: +{(best_accuracy - 0.3333)*100:.2f} percentage points")
