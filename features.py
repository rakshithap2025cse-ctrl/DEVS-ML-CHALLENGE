"""Shared, leakage-safe feature engineering for next-interval energy forecasting.

Every feature for row t uses only information measured at or before time t.
Lags are looked up BY TIMESTAMP (not by row position), so gaps, shuffled rows
or a short input frame never silently pick the wrong neighbour row.
"""
import numpy as np
import pandas as pd

TARGET = "next_interval_Usage_kWh"
DATE_FMT = "%d/%m/%Y %H:%M"
STEP = pd.Timedelta(minutes=15)

RAW_NUMERIC = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]
RAW_INPUT_COLUMNS = ["date"] + RAW_NUMERIC + ["WeekStatus", "Day_of_week", "Load_Type"]

LOAD_MAP = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
DOW_MAP = {d: i for i, d in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}

# lag k (in 15-min steps) of Usage_kWh.  95 / 671 = the *next* slot one day / one week ago.
LAGS = [1, 2, 3, 4, 8, 95, 96, 671, 672]
ROLL_WINDOWS = [4, 16, 96]


def true_timestamp(date_col: pd.Series, nsm: pd.Series) -> pd.Series:
    """Parse DD/MM/YYYY HH:MM and fix the dataset quirk: the 24:00 reading
    (NSM == 0) is stamped 00:00 of the day it CLOSES, so it really belongs to the next day."""
    ts = pd.to_datetime(date_col, format=DATE_FMT, errors="coerce")
    midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (nsm.astype(float) == 0)
    return ts + pd.to_timedelta(midnight.astype(int), unit="D")


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["load_type_code"] = df["Load_Type"].map(LOAD_MAP)
    out["is_weekend"] = (df["WeekStatus"] == "Weekend").astype(float)
    out["dow_code"] = df["Day_of_week"].map(DOW_MAP)
    ang = 2 * np.pi * out["NSM"] / 86400.0
    out["tod_sin"], out["tod_cos"] = np.sin(ang), np.cos(ang)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow_code"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow_code"] / 7)
    return out


BASIC_COLUMNS = list(_basic_features(pd.DataFrame({
    **{c: [0.0] for c in RAW_NUMERIC}, "Load_Type": ["Light_Load"],
    "WeekStatus": ["Weekday"], "Day_of_week": ["Monday"]})).columns)


def build_features(df: pd.DataFrame, history: pd.DataFrame | None = None, engineered: bool = True):
    """Return (X, ts) with X aligned to df's original row order.

    history: optional EARLIER rows (same schema) used only as a source of past Usage_kWh.
    """
    df = df.reset_index(drop=True)
    X = _basic_features(df)
    ts = true_timestamp(df["date"], df["NSM"])
    if not engineered:
        return X, ts

    usage_in = pd.Series(pd.to_numeric(df["Usage_kWh"], errors="coerce").values, index=ts.values)
    usage_in = usage_in[~pd.isna(usage_in.index)]
    parts = [usage_in]
    if history is not None and len(history):
        h = history.reset_index(drop=True)
        hs = pd.Series(pd.to_numeric(h["Usage_kWh"], errors="coerce").values,
                       index=true_timestamp(h["date"], h["NSM"]).values)
        parts = [hs[~pd.isna(hs.index)], usage_in]          # input rows win on overlap
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq=STEP)
    g = s.reindex(grid)                                     # explicit 15-min grid; gaps -> NaN

    idx = pd.DatetimeIndex(ts)
    cur = g.reindex(idx).values
    for k in LAGS:
        X[f"lag_{k}"] = g.reindex(idx - k * STEP).values
    X["diff_1"] = cur - X["lag_1"].values
    X["diff_4"] = cur - X["lag_4"].values
    for w in ROLL_WINDOWS:                                  # window ends AT t (inclusive), never later
        r = g.rolling(w, min_periods=max(2, w // 2))
        X[f"roll_mean_{w}"] = r.mean().reindex(idx).values
        X[f"roll_std_{w}"] = r.std().reindex(idx).values
    # same wall-clock slot, next interval, yesterday / last week (both fully in the past)
    X["usage_vs_yesterday_slot"] = cur - X["lag_95"].values
    return X, ts


# Final feature set (chosen with time-series CV on the training period; see model_training.ipynb).
# Deliberately excludes the 1-day / 1-week lags: they did not help and would need 7 days of history.
SELECTED_FEATURES = BASIC_COLUMNS + [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "diff_1", "diff_4",
    "roll_mean_4", "roll_std_4", "roll_mean_16", "roll_std_16", "roll_mean_96", "roll_std_96",
]
HISTORY_ROWS = 200   # past rows stored in the artifact so a test file that starts right after train has context
