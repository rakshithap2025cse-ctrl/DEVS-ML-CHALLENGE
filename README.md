# MODEL ARENA — Track 2: Steel Plant Energy Consumption Forecasting

Forecast electricity usage (kWh) for the **next 15-minute interval** (`next_interval_Usage_kWh`) from current and past plant telemetry.
Metrics (per the problem statement): **RMSE** and **MAE**, on a later chronological period.

## Result (chronological validation, latest 20 % of `train.csv`)

| Model | RMSE (kWh) | MAE (kWh) |
|---|---|---|
| Persistence baseline (predict "same as now") | 11.945 | 5.277 |
| **Final: LightGBM + lag/rolling features** | **8.189** | **3.811** |

Validation period: 22 Aug – 19 Oct 2018 (5,607 rows), never used for fitting in this evaluation. Full comparison in `model_training.ipynb`.

## Libraries
Python 3.10+, pandas, numpy, scikit-learn, lightgbm, joblib (+ matplotlib, xgboost, jupyter for the notebooks). Exact tested versions: `requirements.txt`.

## Files

| File | Purpose |
|---|---|
| `train_cleaned.csv` | Cleaned dataset: original columns + `timestamp_corrected` (created by `clean_data.py`) |
| `clean_data.py` | Reproduces the cleaned dataset and runs the data checks |
| `eda.ipynb` | Data audit, date-quirk fix, target alignment check, patterns, leakage review |
| `model_training.ipynb` | Split, baselines, 6-model comparison, feature ablation, tuning, ensemble check, interpretation |
| `features.py` | Shared feature engineering (timestamp fix, lags, rolling stats) — used by training **and** inference |
| `train_model.py` | Reproducible training: 80/20 chronological evaluation, then refit on all rows → `model.joblib` |
| `inference.py` | `predict(df)` → one prediction per row, same order |
| `test_pipeline.py` | 8 end-to-end checks (see below) |
| `model.joblib` | Saved pipeline (imputer + LightGBM), feature list, last 200 training rows as past context |
| `validation_metrics.json`, `results/` | Numbers quoted in the write-ups |
| `methodology_summary.md` | 1–2 page summary for the organisers |
| `JUDGE_GUIDE.md` | Personal preparation guide for the judging round |

## Setup and run

```bash
pip install -r requirements.txt
python clean_data.py --train train.csv --out train_cleaned.csv  # cleaned dataset
python train_model.py --train train.csv --out model.joblib   # ~2 s; prints validation metrics
python test_pipeline.py                                      # 8/8 checks (takes ~10 s)
```

## Predict

```python
import pandas as pd
from inference import predict

test = pd.read_csv("test.csv")        # same columns as train.csv; target column not needed
preds = predict(test)                 # numpy array, len(preds) == len(test), same row order
```

## How inference treats time (important)

* Features for a row use **only that row and earlier timestamps**. Later rows are never read, because the label of row *t* equals row *t+1*'s `Usage_kWh` in this dataset — using it would be cheating (checked by `test_04_causality_no_future_rows_used`).
* Past values are found **by timestamp**, so shuffled rows or gaps cannot pick the wrong neighbour.
* The model file stores the last 200 training rows as extra past context, so the first rows of a test file that starts right after `train.csv` still have lag values. Disable with `predict(df, use_bundled_history=False)` or supply your own with `history=`.
* The `date` column has a quirk: the 24:00 reading is stamped `00:00` of the *same* day (`NSM = 0`). `features.true_timestamp` moves it to the next day.

## Tests actually run (`python test_pipeline.py` → 8/8 passed)

1. artifact reloads from disk · 2. batch length, finiteness, order invariance (reversed input gives same answers) · 3. target column not required and not used · 4. causality (removing future rows doesn't change earlier predictions) · 5. single row, NaNs, unseen category, unparseable date · 6. clear error on missing column · 7. RMSE/MAE correctly computed and finite · 8. simulated hidden test: train on first 80 % → predict last 20 % with target removed (RMSE 8.177, MAE 3.796).

## Known limits

See `methodology_summary.md`. In short: only Jan–Oct 2018 is available; peaks and sudden start-up/shutdown moments are under-forecast; accuracy on non-contiguous test rows (missing lags) was not measured.
