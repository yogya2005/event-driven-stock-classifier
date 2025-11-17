# Event-Driven Stock Impact Classifier - Project Complete

## ✅ All Tasks Successfully Completed

### Task 1: Load and Filter FNSPID Dataset ✓
- Downloaded 13+ million rows from HuggingFace FNSPID dataset
- Filtered for 2020 data: 201,557 rows
- Sampled to 20,000 rows for manageable processing
- **Output**: `fnspid_2020_cleaned.csv` (2.2 MB)

### Task 2: Enrich with Stock Data ✓
- Downloaded data for 3,213 unique stocks from yfinance
- Successfully retrieved data for 2,391 stocks (74% success rate)
- Calculated next-day returns for 15,875 examples
- Added stock metadata (sector, market cap, beta)
- Created labels using ±1.5% threshold
- **Final dataset**: 13,568 labeled examples with balanced distribution:
  - Negative (-1): 4,462 (32.9%)
  - Neutral (0): 4,949 (36.5%)
  - Positive (+1): 4,157 (30.6%)
- **Output**: `stock_news_labeled.csv` (2.3 MB)

### Task 3: Feature Engineering ✓
- Created TF-IDF features from article titles (100 features)
- One-hot encoded 11 sectors
- Normalized continuous features (market cap, beta)
- **Total features**: 113
- Train/test split: 10,854 / 2,714 (80/20)
- **Outputs**: 
  - `train_test_split.npz` (1.3 MB)
  - `tfidf_vectorizer.pkl` (4.4 KB)
  - `feature_scaler.pkl` (604 B)

### Task 4: Train and Evaluate Models ✓
**Results (Test Accuracy):**

| Model | Accuracy | Improvement over Baseline |
|-------|----------|---------------------------|
| Random Baseline | 33.33% | - |
| Logistic Regression | 42.23% | +8.90 pp |
| Random Forest | **43.81%** | **+10.48 pp** |
| XGBoost | 43.33% | +10.00 pp |

**🏆 Best Model: Random Forest with 43.81% accuracy**

All models significantly outperform the random baseline, meeting the project target of 40-60% accuracy.

**Model Details:**
- All three models show similar patterns: better at predicting neutral movements
- Precision/recall relatively balanced across classes
- Training times: LR (0.11s), RF (0.15s), XGBoost (0.53s)

**Outputs**: 
- `logistic_regression_model.pkl` (3.4 KB)
- `random_forest_model.pkl` (9.2 MB)
- `xgboost_model.pkl` (745 KB)
- `model_results.pkl` (91 B)

### Task 5: Create Visualizations ✓
Generated four visualizations for milestone report:
1. **Model Comparison Bar Chart** - Shows all model accuracies vs baseline
2. **Confusion Matrices** - Three heatmaps showing prediction patterns
3. **Label Distribution** - Dataset balance visualization
4. **Dataset Sample Table** - First 10 rows with all features

**Outputs**: 
- `model_comparison.png` (123 KB)
- `confusion_matrices.png` (219 KB)
- `label_distribution.png` (117 KB)
- `dataset_sample.png` (336 KB)

---

## 📊 Key Findings

### Dataset Characteristics
- 13,568 total examples across 11 sectors
- Well-balanced three-way classification
- Data from first half of 2020 (Jan-Jun)
- Covers 2,391 unique stocks

### Model Performance
- All models beat baseline by 9-10 percentage points
- Random Forest slightly edges out XGBoost
- Models tend to predict "Neutral" more often (highest recall for that class)
- Positive movements are hardest to predict (lowest recall)

### Challenges Encountered
1. **Data availability**: ~26% of stocks failed to download (delisted or incorrect symbols)
2. **Missing metadata**: ~15% filtered due to incomplete sector/market cap/beta
3. **Dataset format**: Had to use direct CSV loading instead of HuggingFace datasets API due to parsing issues

---

## 📁 Complete File Inventory

### Data Files (3)
- `fnspid_2020_cleaned.csv` - Cleaned 2020 news data
- `stock_news_labeled.csv` - Final labeled dataset with all features
- `train_test_split.npz` - Train/test split

### Model Files (6)
- `tfidf_vectorizer.pkl` - TF-IDF transformer
- `feature_scaler.pkl` - Feature normalizer
- `logistic_regression_model.pkl` - Trained LR model
- `random_forest_model.pkl` - Trained RF model (best)
- `xgboost_model.pkl` - Trained XGBoost model
- `model_results.pkl` - Accuracy results summary

### Visualization Files (4)
- `model_comparison.png`
- `confusion_matrices.png`
- `label_distribution.png`
- `dataset_sample.png`

### Script Files (6)
- `task1_load_filter_data.py`
- `task2_enrich_stock_data.py`
- `task3_feature_engineering.py`
- `task4_train_models.py`
- `task5_create_visualizations.py`
- `explore_dataset.py`

---

## 🎯 Next Steps for Milestone Report

### Required Sections

1. **Brief Project Recap** (2-3 sentences)
   - Multi-class stock reaction prediction from news
   - Three models: LR, RF, XGBoost
   - ±1.5% threshold for three-way classification

2. **Significant Accomplishments**
   - Dataset creation: 13,568 labeled examples
   - Feature engineering: 113 combined features
   - Model training: 43.81% best accuracy (vs 33% baseline)
   - Include visualizations as proof

3. **Challenges or Roadblocks**
   - Stock data availability (~26% failed)
   - Weekend/holiday date handling
   - Dataset parsing issues with HuggingFace loader

4. **Changes from Original Plan**
   - Used individual stock-news pairs instead of "major market events"
   - More systematic and scalable approach
   - Broader event diversity

---

## 🚀 Potential Improvements (Future Work)

1. **Feature Enhancement**
   - Sentiment analysis on article titles
   - Include article publication time
   - Add momentum/volatility features
   - Sector-specific thresholds

2. **Model Optimization**
   - Hyperparameter tuning (GridSearchCV)
   - Ensemble methods
   - Neural network approaches
   - Class imbalance techniques (SMOTE)

3. **Data Expansion**
   - Full year 2020 data
   - Multiple years for temporal patterns
   - Include article full text (not just titles)
   - More granular time windows (intraday)

---

## 📈 Project Success Criteria

✅ **Dataset**: Successfully created labeled dataset with >10,000 examples  
✅ **Features**: Implemented multi-source feature engineering (text + numerical)  
✅ **Models**: Trained and evaluated 3 different model types  
✅ **Performance**: Achieved 40-60% target accuracy range  
✅ **Visualizations**: Created publication-ready figures for report  
✅ **Documentation**: Complete with code, data, and results  

---

**Project Status**: ✅ COMPLETE  
**Total Runtime**: ~15-20 minutes  
**Final Accuracy**: 43.81% (Random Forest)  
**Target Range**: 40-60% ✓  

All files are ready for inclusion in the Milestone 2 report.
