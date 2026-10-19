"""Reproducible training script.

Usage:  python train_model.py --train train.csv --out model.joblib

1. Chronological 80/20 split -> honest validation scores (baselines + LightGBM).
2. Refit the SAME configuration on all labelled rows and save model.joblib
   (the hidden test period comes after the training data, so the latest rows are the most useful).
"""
import argparse, json, time
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from features import TARGET, SELECTED_FEATURES, HISTORY_ROWS, RAW_INPUT_COLUMNS, build_features

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mae(y, p): return float(mean_absolute_error(y, p))
def make_model(): return make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMRegressor(**LGBM_PARAMS))


def main(train_path, out_path, metrics_path):
    df = pd.read_csv(train_path)
    missing = [c for c in RAW_INPUT_COLUMNS + [TARGET] if c not in df.columns]
    assert not missing, f"train.csv is missing columns: {missing}"
    y = df[TARGET].values
    X, ts = build_features(df)                       # file order is chronological (verified in EDA)
    assert ts.is_monotonic_increasing, "rows must be in chronological order after timestamp fix"
    X = X[SELECTED_FEATURES]

    n = len(df); split = int(n * 0.8)
    tr, va = np.arange(0, split - 1), np.arange(split, n)     # 1-row gap: last train label == first val Usage
    t0 = time.time(); m = make_model().fit(X.iloc[tr], y[tr]); secs = time.time() - t0
    pv, pt = m.predict(X.iloc[va]), m.predict(X.iloc[tr])
    persist = df["Usage_kWh"].values
    metrics = dict(
        train_rows=len(tr), val_rows=len(va), train_period=[str(ts[tr[0]]), str(ts[tr[-1]])],
        val_period=[str(ts[va[0]]), str(ts[va[-1]])], fit_seconds=round(secs, 2),
        val_RMSE=rmse(y[va], pv), val_MAE=mae(y[va], pv), train_RMSE=rmse(y[tr], pt), train_MAE=mae(y[tr], pt),
        persistence_val_RMSE=rmse(y[va], persist[va]), persistence_val_MAE=mae(y[va], persist[va]),
        val_target_mean=float(y[va].mean()))
    assert len(pv) == len(va) and np.isfinite(pv).all()
    print(json.dumps(metrics, indent=2))

    final = make_model().fit(X, y)                    # refit on every labelled row
    artifact = dict(model=final, feature_columns=SELECTED_FEATURES, lgbm_params=LGBM_PARAMS,
                    history=df[RAW_INPUT_COLUMNS].tail(HISTORY_ROWS).reset_index(drop=True),
                    validation_metrics=metrics, trained_on_rows=n)
    joblib.dump(artifact, out_path)
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"saved {out_path} (trained on {n} rows) and {metrics_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv"); ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--metrics", default="validation_metrics.json")
    a = ap.parse_args(); main(a.train, a.out, a.metrics)
