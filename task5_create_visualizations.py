"""
Task 5: Create Visualizations
Generate visualizations for the milestone report.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from scipy import sparse

print("=" * 60)
print("TASK 5: CREATE VISUALIZATIONS")
print("=" * 60)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# =============================================================================
# 1. MODEL COMPARISON BAR CHART
# =============================================================================
print("\n1. Creating model comparison chart...")

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
print("✓ Saved model_comparison.png")
plt.close()

# =============================================================================
# 2. CONFUSION MATRICES
# =============================================================================
print("\n2. Creating confusion matrices...")

# Load data
data = np.load('train_test_split.npz', allow_pickle=True)
y_test = data['y_test']
X_test = data['X_test']

# Handle sparse matrix
if isinstance(X_test, np.ndarray) and X_test.dtype == object:
    X_test = X_test.item()

# Load models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Get predictions
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
    axes[idx].set_title(f'{name}\\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrices.png")
plt.close()

# =============================================================================
# 3. LABEL DISTRIBUTION
# =============================================================================
print("\n3. Creating label distribution chart...")

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
             f'{val}\\n({val/sum(label_values)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved label_distribution.png")
plt.close()

# =============================================================================
# 4. DATASET SAMPLE TABLE
# =============================================================================
print("\n4. Creating dataset sample table...")

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
print("✓ Saved dataset_sample.png")
plt.close()

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated files:")
print("  1. model_comparison.png")
print("  2. confusion_matrices.png")
print("  3. label_distribution.png")
print("  4. dataset_sample.png")
