# 📩 SMS Spam Detection — Decision-Ready Pipeline

Dual **TF-IDF (word + char)** → **Linear SVM (calibrated)** → nested CV + randomized search → threshold policy → explainability & robustness checks.

**Full case study:** `CASE_STUDY.md`

---

## ✅ What this repo provides
- **Leak-safe evaluation** (nested CV)
- **Calibrated probabilities** for decision-making
- **Explicit threshold policy** (exported & reused in inference)
- **Exported artifacts** under `./artifacts/`
- **CLI inference** via `predict.py`
- **Minimal dependencies**, CPU-friendly pipeline

---

## 📂 Dataset
- **Source**: UCI SMS Spam Collection (Kaggle mirror)
- **Columns**:
  - `v1` — original label (`ham` or `spam`)
  - `v2` — raw SMS text
  - `label` — normalized label (0 = ham, 1 = spam)
  - `text` — cleaned SMS text after preprocessing

### Local (recommended)
1) Download the CSV from Kaggle.
2) Place it here:

`./data/raw/SPAM text message 20170820 - Data.csv`

Dataset files are not included in this repository.

### Kaggle
Default Kaggle path used in the notebook:

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
   - Threshold tuning (F1-optimized by default)
8. **Evaluation** (classification report, PR/ROC curves, calibration plots)
9. **Explainability** (top spam/ham n-grams, FP/FN cases)
10. **Robustness** (obfuscation stress test)
11. **Artifacts export** (model, metrics, metadata)

---

## 📈 Results
Run-specific metrics and plots are exported under `./artifacts/`:
- Metrics: `artifacts/metrics.json`
- Configuration + threshold: `artifacts/metadata.json`

---

## 🛠️ Environment
- **Python**: 3.10–3.12

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/tarekmasryo/sms-spam-detection
cd sms-spam-detection
pip install -r requirements.txt
```

Place the dataset CSV under:

`./data/raw/SPAM text message 20170820 - Data.csv`

Then run:

```bash
jupyter notebook sms-spam-detection.ipynb
```

---

## 🔮 CLI Inference (after running the notebook)

Run the notebook once to generate artifacts under `./artifacts/`, then:

```bash
python predict.py --text "win a free prize now"
python predict.py "See you at 6?"
```

---

## 🔍 Notes on Methodology
- **No leakage**: vectorizers fit only on training folds.
- **Nested CV**: outer folds provide unbiased performance.
- **Calibrated SVC**: converts margin scores → reliable probabilities.
- **Threshold policy**: exported for consistent inference.
- **Robustness**: includes an obfuscation stress test; TF-IDF typically weakens under heavy adversarial transforms.

---

## 📜 License
MIT (code) — dataset subject to original UCI license.
