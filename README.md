# 📩 SMS Spam Detection — End-to-End Pipeline

Dual **TF-IDF (word + char)** → **Linear SVM (calibrated)** → nested CV + randomized search → F1-optimized thresholding → explainability & robustness tests.

---

## 🚀 Why this notebook?
- **Kaggle-ready**: clean imports, reproducible seeds, minimal dependencies.  
- **EDA**: class balance, message length distributions.  
- **Modeling**: dual TF-IDF features + Linear SVC baseline.  
- **Model selection**: nested CV + randomized search for hyperparameters.  
- **Calibration + threshold tuning**: sigmoid scaling → reliable probabilities.  
- **Explainability**: top spam/ham n-grams.  
- **Robustness**: obfuscation test (`fr33`, `w1n`, `0ffers`).  
- **Artifacts**: models, metrics, metadata, and figures exported to `./artifacts/`.

---

## 📂 Dataset
- **Source**: [UCI SMS Spam Collection (Kaggle mirror)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **Columns**:
  - `v1` — original label (`ham` or `spam`)  
  - `v2` — raw SMS text  
  - `label` — normalized label (0 = ham, 1 = spam)  
  - `text` — cleaned SMS text after preprocessing  

> Default Kaggle path:  
`/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv`

---

## 🧱 Notebook Outline
1. **Setup & Imports**  
2. **Load & Audit** (column normalization, missing/duplicates removal)  
3. **EDA — Distributions** (class balance, message length histograms)  
4. **Text Normalization** (URLs, numbers, emails → placeholders)  
5. **Train/Test Split** (stratified)  
6. **Baseline model** (Logistic Regression)  
7. **Dual TF-IDF + Linear SVC Pipeline**  
   - Nested CV + randomized search  
   - Probability calibration (Platt scaling)  
   - Threshold tuning (F1-optimized)  
8. **Evaluation** (classification report, PR/ROC curves, calibration plots)  
9. **Explainability** (top spam/ham n-grams, FP/FN cases)  
10. **Robustness** (obfuscation stress test)  
11. **Artifacts export** (models, metrics, metadata JSON)

---

## 📈 Results (Hold-out Set)
- **F1 ≈ 0.95**  
- **PR-AUC ≈ 0.98**  
- **ROC-AUC ≈ 0.99**  
- **Brier ≈ 0.01**  

Confusion matrix and evaluation curves are saved under `./artifacts/`.

---

## 🛠️ Environment
- **Python**: 3.10–3.12  
- **Core**: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`  

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/tarekmasryo/sms-spam-detection
cd sms-spam-detection
jupyter notebook notebooks/sms_spam_pipeline.ipynb
```

- Place `SPAM text message 20170820 - Data.csv` under the repo or attach on Kaggle.

---

## 🔍 Notes on Methodology
- **No leakage**: vectorizers fit only on training folds.  
- **Nested CV**: outer folds provide unbiased performance.  
- **Calibrated SVC**: converts margin scores → reliable probabilities.  
- **Threshold tuning**: optimizes F1 for imbalanced spam/ham classes.  
- **Robustness**: ~10% F1 drop under obfuscation → TF-IDF weakness.  
- **Next steps**: add augmented obfuscated data, compare with **FastText/DistilBERT**.

---

## 📜 License
MIT (code) — dataset subject to original UCI license.

