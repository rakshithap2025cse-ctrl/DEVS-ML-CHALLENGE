"""Inference for next-interval (t+1) electricity usage.

    from inference import predict
    preds = predict(df)            # df has the train.csv columns (target column NOT needed)

Rules the function follows
- Output: 1-D numpy array, one finite float per input row, SAME order as the input rows.
- Causal: features for row t use only rows at or before time t. Later rows are never read,
  so the target of row t is never inferred from row t+1's Usage_kWh.
- Lags/rolling values are looked up by TIMESTAMP, so shuffled rows or gaps cannot pick the wrong neighbour.
- History: the artifact stores the last 200 training rows. They are used only as extra *past* context
  (helpful when the test file starts right after train.csv). Pass history=... to override, or
  use_bundled_history=False to disable.
- Nothing is refitted at inference time; imputation values live inside the saved pipeline.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so "import features" works from any folder
from features import RAW_INPUT_COLUMNS, TARGET, build_features

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_HERE, "model.joblib")   # always the file next to this script
_CACHE = {}


def load_artifact(path=DEFAULT_MODEL):
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def predict(df: pd.DataFrame, model_path=DEFAULT_MODEL, history: pd.DataFrame | None = None,
            use_bundled_history: bool = True) -> np.ndarray:
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    art = load_artifact(model_path)
    if history is None and use_bundled_history:
        history = art["history"]
    X, _ = build_features(df[RAW_INPUT_COLUMNS], history=history)
    X = X.reindex(columns=art["feature_columns"])           # exact training column order
    pred = np.asarray(art["model"].predict(X), dtype=float)
    # energy cannot be negative; also guarantees finite output
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return pred


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    out = predict(pd.read_csv(src))
    print(f"{len(out)} predictions; first 5: {np.round(out[:5], 2)}")
