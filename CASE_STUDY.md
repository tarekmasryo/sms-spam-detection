# Case Study — SMS Spam Detection (Decision-Ready Pipeline)

## Overview
This project builds a **CPU-friendly** SMS spam classifier that outputs **calibrated probabilities** and applies an explicit **threshold policy** for production-like decision control. It ships with **exported artifacts**, a **CLI predictor**, and a **CI smoke test** to keep the pipeline runnable.

## Problem
Spam detection is asymmetric-risk classification:
- **False positives** block legitimate messages (high user cost).
- **False negatives** let scams through (safety/abuse cost).
We need a solution that is **fast**, **reproducible**, and **auditable**.

## Approach (and why this choice)
- **Dual TF-IDF (word + char n-grams)** to capture both phrasing and mild obfuscations.
- **Linear SVM** for strong performance on sparse text and fast CPU inference.
- **Probability calibration** (sigmoid / Platt scaling) to produce usable probability scores.

This classical pipeline is preferred here over heavier neural models for **simplicity, speed, and low operational cost** while remaining highly competitive on this dataset.

## Evaluation (Leak-Safe)
- Stratified train/test split.
- **Nested CV** for hyperparameter tuning with an unbiased outer estimate.
- Metrics include **F1**, **PR-AUC**, **ROC-AUC**, and **Brier score** (calibration quality).

> Run-specific metrics are exported to `artifacts/metrics.json` and configuration (including the threshold) to `artifacts/metadata.json`.

## Decision Policy (Thresholding)
Instead of defaulting to 0.5, the pipeline selects an operating **threshold** (F1-optimized by default) and exports it for consistent inference.  
This policy can be adjusted to prioritize **precision** (reduce false positives) or **recall** (catch more spam), depending on operating constraints.

## Artifacts
Running the notebook exports under `./artifacts/`:
- `sms_spam_calibrated.joblib` (trained pipeline)
- `metrics.json` (run metrics)
- `metadata.json` (config + selected threshold)

## How to Use
1. `pip install -r requirements.txt`
2. Place the CSV under `data/raw/` (see `data/raw/README.md`)
3. Run `sms-spam-detection.ipynb` to generate artifacts
4. Predict via CLI:
   - `python predict.py --text "win a free prize now"`

## Limitations & Next Steps
TF-IDF is weaker under heavy adversarial obfuscation. Next steps:
- augment with obfuscated variants
- compare against **FastText / DistilBERT**
- consider cost-aware thresholding (explicit FP vs FN costs)
