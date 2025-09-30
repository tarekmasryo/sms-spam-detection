# 📩 SMS Spam Detection — End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange)](https://scikit-learn.org/stable/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Made with ❤️ by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

---

## 📌 Highlights
- **Dual TF-IDF (word + char)** → captures both semantics and character patterns.  
- **Linear SVM + Calibration** → strong classifier with reliable probabilities.  
- **Nested CV + RandomizedSearch** → unbiased model selection.  
- **Threshold tuning** → optimized for **F1 score**.  
- **Explainability** → top spam/ham n-grams exported.  
- **Robustness** → obfuscation stress-test (`fr33`, `w1n`, `0ffers`).  
- **Artifacts** → ready-to-serve (`.joblib` models, metrics, figures).  

**Results (hold-out):**  
F1 ≈ **0.95** · PR AUC ≈ **0.98** · ROC AUC ≈ **0.99** · Brier ≈ **0.01**

---

## 📂 Dataset
| Column  | Description                          |
|---------|--------------------------------------|
| `v1`    | original label (`ham` or `spam`)     |
| `v2`    | raw SMS text                         |
| `label` | normalized label (0 = ham, 1 = spam) |
| `text`  | cleaned SMS text after preprocessing |


---

## ⚡ Quick Inference
```python
import joblib, pandas as pd

pipe = joblib.load("artifacts/sms_spam_calibrated.joblib")
msgs = ["Free entry in 2 a wkly comp!", "See you at 6?"]
probs = pipe.predict_proba(pd.DataFrame({"text": msgs}))[:,1]
preds = (probs >= 0.5).astype(int)
print(list(zip(msgs, preds, probs.round(3))))
```

---

## 🧱 Project Structure
```
sms-spam-detection/
│── notebooks/
│   └── sms_spam_pipeline.ipynb
│── artifacts/
│   ├── sms_spam_linear_svc.joblib
│   ├── sms_spam_calibrated.joblib
│   ├── tfidf_features.joblib
│   ├── metadata.json
│   └── metrics_summary.csv
│── assets/
│   ├── class_balance.png
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── roc_curve.png
│── README.md
```

---

## 🛠️ Environment
- **Python**: 3.10–3.12  
- **Core packages**:  
  `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`

Install:
```bash
pip install -r requirements.txt
```

---

## 🔍 Methodology
- **Preprocessing**: normalize text (URLs → `__url__`, numbers → `__number__`, etc.).  
- **No leakage**: vectorizers fit on training folds only.  
- **Nested CV**: outer folds estimate F1 unbiasedly, inner folds tune hyperparameters.  
- **Calibration**: Platt scaling (sigmoid) for probabilistic outputs.  
- **Threshold tuning**: selected via CV to maximize F1.  
- **Explainability**: top n-grams identified for both spam and ham.  
- **Robustness test**: synthetic obfuscations to test generalization.  

---

## 🚀 Next Steps
- Augment training data with obfuscated variants.  
- Benchmark against **embedding-based models** (FastText, DistilBERT).  
- Wrap model in a **FastAPI endpoint** + optional **Streamlit dashboard**.  

---

## 📜 License
- Code: MIT  
- Dataset: Subject to [UCI SMS Spam Collection license](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

