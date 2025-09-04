# Credit Card Fraud Detection

## Abstract
Credit card fraud undermines financial trust and causes significant losses for consumers and institutions. This project explores how machine learning can detect fraudulent transactions using a real-world, highly imbalanced dataset.

I built two approaches: a classical baseline using engineered features with logistic regression, and a neural model (MLP). The goal was to see how much a simple neural network improves over a strong classical baseline on this task. Both models use the same dataset and evaluation setup for a fair comparison.

## 1. Introduction
Detecting fraud is tricky because fraudulent transactions are rare and often resemble normal activity. Class imbalance, data leakage risks, and distribution drift complicate modeling. I started with a transparent baseline to understand which patterns the model learns before trying a neural approach.

## 2. Related Work
Many academic and industry systems rely on interpretable models (e.g., logistic regression, tree ensembles) due to auditability needs. Techniques like class weighting, resampling (SMOTE), and calibrated thresholds are common. Neural networks can capture nonlinear interactions but require careful regularization and validation to avoid overfitting and leakage.

## 3. Data
The dataset is provided in `data/raw/creditcard.csv` and contains transaction-level records with anonymized features and a binary label (`Class` or `is_fraud`). After cleaning and ensuring no duplicates, the commonly used reference dataset contains **284,807** transactions with roughly **0.17%** labeled as fraud.

I use an 80/20 train/validation split with a fixed random seed to ensure consistent evaluation.

## 4. Methods

### 4.1 Classical Baseline
- Standardization with `StandardScaler`
- Logistic Regression (class-weight balanced)
- Threshold tuning on validation set
- Optional SMOTE experiments (documented but not default)

### 4.2 Neural Baseline (MLP)
- MLP with 1–2 hidden layers, ReLU, dropout
- Binary cross-entropy with class weights
- Early stopping on validation AUROC / F1
- Threshold tuning for precision–recall trade-off

### 4.3 Pipeline
Raw CSV → Clean → Split (train/val) → [Scale, Rebalance?] → [LogReg | MLP] → {Fraud, Legit}

## 5. Experimental Setup
- Fixed 80/20 split with seed 42  
- Metrics: AUROC, PR-AUC, precision, recall, F1 (macro + fraud-class F1)  
- Models saved to `models/` and reports to `reports/`

## 6. Results — Classical Baseline
**Validation performance (fill in):**
- AUROC: **0.XXX**  
- PR-AUC: **0.XXX**  
- F1 (fraud class): **0.XXX**  

Artifacts:
- Confusion matrix: `reports/confusion_matrix_classical.png`  
- Top features (by |coef|): `reports/top_features_classical.txt`  
- Metrics/report: `reports/metrics_classical.json`, `reports/classification_report_classical.txt`

## 7. Results — Neural Baseline (MLP)
**Validation performance (fill in):**
- AUROC: **0.XXX**  
- PR-AUC: **0.XXX**  
- F1 (fraud class): **0.XXX**  

Artifacts:
- Metrics/report: `reports/metrics_mlp.json`, `reports/classification_report_mlp.txt`  
- Sample preds: `reports/sample_val_preds_mlp.csv`

## 8. Error Analysis
Sample predictions:
- Classical: `reports/sample_val_preds_classical.csv`
- MLP: `reports/sample_val_preds_mlp.csv`

## 9. Limitations
- Extreme class imbalance → accuracy is misleading  
- Potential dataset shift vs production  
- Anonymized feature set limits feature engineering  
- Threshold selection depends on business constraints (precision vs recall)

## 10. Reproducibility

**Setup**
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

**Run**
Open `notebooks/fraud_detection.ipynb` and execute cells top-to-bottom.
Models and reports will be saved under `models/` and `reports/`.

## 11. Project Structure
```
.
├── data/
│   └── raw/
│       └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── requirements.txt
└── README.md
```

## Acknowledgements
This repo layout and evaluation approach draw on common industry practices for fraud detection. The widely used dataset originates from the "Credit Card Fraud Detection" dataset.
