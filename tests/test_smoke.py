import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _train_tiny_model() -> Pipeline:
    # Tiny, deterministic dataset (balanced) just to validate the end-to-end plumbing.
    texts = [
        "win a free prize now",
        "claim your cash reward",
        "limited offer, click link",
        "free entry in a weekly draw",
        "urgent! you won a prize",
        "see you at 6",
        "are we still meeting tomorrow?",
        "ok thanks",
        "can you call me later?",
        "let's grab lunch",
    ]
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    X = pd.DataFrame({"text": texts})

    features = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1), "text"),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=200, random_state=42)

    model = Pipeline([("features", features), ("clf", clf)])
    model.fit(X, y)
    return model


def test_predict_cli_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    model = _train_tiny_model()
    model_path = artifacts / "sms_spam_calibrated.joblib"
    joblib.dump(model, model_path)

    metadata_path = artifacts / "metadata.json"
    metadata_path.write_text(json.dumps({"opt_threshold": 0.5}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo_root / "predict.py"),
        "--text",
        "win a free prize now",
        "--artifacts-dir",
        str(artifacts),
        "--json",
    ]

    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    out = json.loads(res.stdout.strip())
    assert "spam_probability" in out
    assert 0.0 <= float(out["spam_probability"]) <= 1.0
    assert out["prediction"] in [0, 1]
    assert out["prediction_label"] in ["ham", "spam"]
