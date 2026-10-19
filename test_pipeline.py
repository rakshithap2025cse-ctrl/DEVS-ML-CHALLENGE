"""End-to-end checks. Run:  python test_pipeline.py   (or: pytest -q test_pipeline.py)"""
import os, tempfile
import numpy as np, pandas as pd, joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import train_model
from features import TARGET, RAW_INPUT_COLUMNS, SELECTED_FEATURES, build_features
from inference import predict, load_artifact

DF = pd.read_csv("train.csv")
ART = "model.joblib"


def test_01_artifact_exists_and_reloads():
    assert os.path.exists(ART)
    joblib.load(ART)                                         # fresh reload from disk, not the cache
    assert load_artifact(ART)["feature_columns"] == SELECTED_FEATURES


def test_02_batch_length_finite_and_order():
    sample = DF.iloc[5000:5300].drop(columns=[TARGET])       # target column NOT supplied (test 8)
    p = predict(sample)
    assert p.shape == (len(sample),) and np.isfinite(p).all()
    # order preserved: prediction for each row must equal the prediction obtained for that row alone with its past
    p_rev = predict(sample.iloc[::-1].reset_index(drop=True))[::-1]   # shuffled/reversed input -> same answers
    assert np.allclose(p, p_rev, atol=1e-9), "row order / timestamp lookup is not order-invariant"


def test_03_target_not_required_and_not_used():
    s = DF.iloc[8000:8100]
    a = predict(s.drop(columns=[TARGET])); b = predict(s.assign(**{TARGET: 12345.0}))
    assert np.allclose(a, b), "predictions changed when the target column changed -> target leaks into inference"


def test_04_causality_no_future_rows_used():
    full = DF.iloc[10000:10200].drop(columns=[TARGET])
    p_full = predict(full)
    p_cut = predict(full.iloc[:120])                          # same rows, but rows 120+ (the 'future') removed
    assert np.allclose(p_full[:120], p_cut, atol=1e-9), "a prediction depends on rows that come later in time"


def test_05_single_row_and_missing_values_and_unseen_category():
    one = DF.iloc[[500]].drop(columns=[TARGET]); assert predict(one).shape == (1,)
    bad = DF.iloc[600:610].drop(columns=[TARGET]).copy()
    bad.loc[bad.index[2], "Lagging_Current_Power_Factor"] = np.nan
    bad.loc[bad.index[3], "Load_Type"] = "Some_New_Load"
    bad.loc[bad.index[4], "Usage_kWh"] = np.nan
    p = predict(bad); assert len(p) == 10 and np.isfinite(p).all()


def test_06_missing_column_raises_clear_error():
    try:
        predict(DF.drop(columns=["NSM"]).head(5)); raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "NSM" in str(e)


def test_07_metrics_are_finite_and_correctly_computed():
    s = DF.iloc[20000:20500]; p = predict(s.drop(columns=[TARGET])); y = s[TARGET].values
    r = np.sqrt(np.mean((y - p) ** 2)); a = np.mean(np.abs(y - p))
    assert np.isclose(r, np.sqrt(mean_squared_error(y, p))) and np.isclose(a, mean_absolute_error(y, p))
    assert np.isfinite(r) and np.isfinite(a)


def test_08_simulated_hidden_test_end_to_end():
    """Train on the first 80% only (fresh artifact), then predict the last 20% with the target removed."""
    with tempfile.TemporaryDirectory() as d:
        cut = int(len(DF) * 0.8)
        head, tail = DF.iloc[:cut], DF.iloc[cut:]
        head.to_csv(f"{d}/train80.csv", index=False)
        train_model.main(f"{d}/train80.csv", f"{d}/m.joblib", f"{d}/m.json")
        p = predict(tail.drop(columns=[TARGET]).reset_index(drop=True), model_path=f"{d}/m.joblib")
        y = tail[TARGET].values
        rm, ma = float(np.sqrt(mean_squared_error(y, p))), float(mean_absolute_error(y, p))
        persist = float(np.sqrt(mean_squared_error(y, tail["Usage_kWh"].values)))
        print(f"simulated hidden test: n={len(p)} RMSE={rm:.3f} MAE={ma:.3f} (persistence RMSE {persist:.3f})")
        assert len(p) == len(tail) and np.isfinite(p).all() and rm < persist


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)}/{len(fns)} tests passed")
