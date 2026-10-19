# Methodology Summary — Track 2: Steel Plant Energy Forecasting

**Task.** Forecast `next_interval_Usage_kWh` (electricity in the next 15 minutes) from current and past telemetry. Metrics: RMSE and MAE on a later period.

## 1. Key findings from EDA
* 28,031 rows, 15-minute readings, 1 Jan – 19 Oct 2018. No missing values, no duplicates.
* **Date problem found and fixed:** the 24:00 reading (`NSM = 0`, 291 rows) is stamped `00:00` of the day it *closes*, so `date` jumps backwards 291 times. Corrected timestamps are strictly increasing with exact 15-minute spacing.
* **Target alignment verified:** `next_interval_Usage_kWh` equals the next row's `Usage_kWh` in 100 % of consecutive rows. We use the given label unchanged — and therefore must never read the next row at inference (that would reveal the answer).
* Consumption is **bimodal**: idle/light load (~3–5 kWh) vs production (50–80+ kWh). Weekday mean 34.2 kWh vs weekend 11.9 (Sunday 7.4). Monthly mean falls from 42 kWh (Jan) to 20 kWh (Sep).
* Usage is strongly autocorrelated at 15 min (0.91) but only moderately at 1 day (0.60) / 1 week (0.67); a "same time yesterday" forecast is far worse than "same as now". `CO2(tCO2)` is essentially derived from current usage.
* High values (271 rows above the IQR fence) are genuine production peaks and were kept.

## 2. Validation approach
Chronological: first 80 % of rows for training (to 22 Aug 2018, 22,423 rows), last 20 % for validation (22 Aug – 19 Oct, 5,607 rows), one row dropped between them. No random splits. Imputation values are learned on training rows only. Model selection and tuning used `TimeSeriesSplit` (5 expanding folds) **inside the training period**; the validation period was used for reporting. No validation rows were dropped (all engineered features available).

## 3. Features and leakage control
Inputs: given measurements, cyclic time-of-day and weekday encodings, ordinal `Load_Type`, weekend flag. Engineered from **past** values only: usage lags 1, 2, 3, 4, 8 steps, first differences, and rolling mean/std ending at time *t* over 1 h, 4 h, 24 h. Lags are looked up by timestamp. The target and any later row are never used; a test shows predictions for earlier rows are unchanged when later rows are deleted. Daily/weekly lags were tested and not kept (no gain, would need 7 days of history).

## 4. Models and validation scores (RMSE / MAE, kWh)
| Model | RMSE | MAE |
|---|---|---|
| Persistence (predict current usage) | 11.945 | 5.277 |
| Seasonal naive (same slot yesterday / last week) | 26.689 / 23.578 | 13.813 / 12.193 |
| Ridge (raw → engineered) | 11.455 → 10.347 | 6.361 → 5.804 |
| Random Forest (raw → engineered) | 8.876 → 8.411 | 4.050 → 3.869 |
| Extra Trees (raw → engineered) | 8.961 → 8.483 | 4.125 → 3.923 |
| HistGradientBoosting (raw → engineered) | 8.634 → 8.346 | 3.989 → 3.985 |
| XGBoost (raw → engineered) | 8.753 → 8.295 | 4.108 → 3.950 |
| LightGBM (raw → engineered, 32 features) | 8.652 → 8.264 | 3.969 → 3.873 |
| **Final: LightGBM, selected 27 features** | **8.189** | **3.811** |

Final model vs persistence: RMSE −31 %, MAE −28 %; validation R² 0.924. Train RMSE 7.12 vs validation 8.19: a modest gap (Random Forest/Extra Trees show a far larger one, ~4–5 vs ~8.4). Tuning (25 random-search trials, time-series CV) improved CV RMSE only 10.245 → 10.182 (fold std ≈ 1.2) and averaging four models changed validation RMSE by ~0.03, so the final model is a single LightGBM with default-style settings, refit on all 28,031 labelled rows for the saved artifact. Differences among the near-tied feature sets (validation RMSE 8.21–8.29) are within noise; the selected set was a tie-break favouring a short (24 h) history requirement. A simulated hidden test (train on 80 %, predict the remaining 20 % without labels through `inference.py`) gave RMSE 8.177 / MAE 3.796.

## 5. Business takeaways and interpretability
* Typical error is **~3.8 kWh per interval (16 % of the mean 23.2 kWh)**. Accuracy is excellent in light load (MAE 1.3 kWh) and weekends (1.9), weaker in Maximum_Load (8.5).
* Permutation importance (what the model relies on, **not** causal): current `Usage_kWh` dominates (RMSE +19.8 kWh when shuffled), then time of day (+2.8), `CO2` (+1.1, a proxy for usage), lagging reactive power (+0.8), and the last change in usage (+0.7).
* The model is best used as a **next-interval early warning of the plant's operating level**, letting operators defer secondary loads before a high-demand interval.
* **Limitation for tariff protection:** in the top 10 % of intervals (≥ 62.5 kWh) the model under-forecasts by ~10 kWh on average (MAE 15.2), and at sudden start-up/shutdown moments (intervals changing by > 20 kWh, 9 % of rows) MAE is 17.9 vs 2.4 elsewhere. A safety margin or an upper-quantile forecast would be needed before using it to guard a demand threshold; this was not built or evaluated.
* **Data limits:** one plant, Jan–Oct 2018 only; no weather, production schedule or equipment status; validation (Aug–Oct) was a lower-load period than training (mean 23.2 vs 29.1 kWh), and the hidden test period may contain seasons never seen in training. Accuracy when the test rows are non-contiguous (missing lags) was not measured.
