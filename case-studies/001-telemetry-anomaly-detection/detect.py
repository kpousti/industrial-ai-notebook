"""Case study 001: catching failing equipment early from sensor telemetry.

Compares two detectors on synthetic rotating-equipment data:
  1. Rolling z-score  - simple, explainable baseline per sensor
  2. Isolation Forest - multivariate, catches combinations a single sensor misses

All data is synthetic. Run:  python detect.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

SENSORS = ["vibration_mm_s", "bearing_temp_c", "exhaust_temp_c"]


def make_telemetry(n_hours: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Generate hourly telemetry with injected faults and ground-truth labels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # Normal behavior: a daily load cycle plus sensor noise
    load = 0.5 + 0.3 * np.sin(2 * np.pi * t / 24)
    df = pd.DataFrame({
        "vibration_mm_s": 2.0 + 0.8 * load + rng.normal(0, 0.15, n_hours),
        "bearing_temp_c": 70 + 10 * load + rng.normal(0, 1.0, n_hours),
        "exhaust_temp_c": 480 + 40 * load + rng.normal(0, 4.0, n_hours),
    })
    df["is_fault"] = 0

    # Fault type A: sudden vibration spikes (e.g. a loose component)
    spikes = rng.choice(np.arange(200, n_hours), size=12, replace=False)
    df.loc[spikes, "vibration_mm_s"] += rng.uniform(1.5, 3.0, size=12)
    df.loc[spikes, "is_fault"] = 1

    # Fault type B: slow bearing degradation, temp AND vibration creep together
    start, end = 1500, 1560
    ramp = np.linspace(0, 1, end - start)
    df.loc[start:end - 1, "bearing_temp_c"] += 6 * ramp
    df.loc[start:end - 1, "vibration_mm_s"] += 0.6 * ramp
    df.loc[start + 20:end - 1, "is_fault"] = 1  # label once it's meaningful

    return df


def rolling_zscore_flags(df: pd.DataFrame, window: int = 48, threshold: float = 3.5) -> pd.Series:
    """Flag a row if any sensor sits > threshold std devs from its trailing mean."""
    flags = pd.Series(False, index=df.index)
    for col in SENSORS:
        # shift(1) so a point is judged only against the past, never itself
        mean = df[col].rolling(window).mean().shift(1)
        std = df[col].rolling(window).std().shift(1)
        z = (df[col] - mean) / std
        flags |= z.abs() > threshold
    return flags


def isolation_forest_flags(df: pd.DataFrame, train_hours: int = 500, seed: int = 7) -> pd.Series:
    """Train on an early window assumed healthy, then score every row."""
    feats = df[SENSORS].copy()
    # Ratio features expose drift relative to load (temp rising while vibration rises)
    feats["temp_per_vib"] = feats["bearing_temp_c"] / feats["vibration_mm_s"]
    model = IsolationForest(contamination=0.01, random_state=seed)
    model.fit(feats.iloc[:train_hours])
    return pd.Series(model.predict(feats) == -1, index=df.index)


def score(df: pd.DataFrame, flags: pd.Series) -> dict:
    """Precision/recall against labels, ignoring the warm-up window."""
    mask = df.index >= 500
    y_true, y_pred = df.loc[mask, "is_fault"], flags[mask].astype(int)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
        "alerts": int(y_pred.sum()),
    }


if __name__ == "__main__":
    data = make_telemetry()
    results = {
        "rolling z-score": score(data, rolling_zscore_flags(data)),
        "isolation forest": score(data, isolation_forest_flags(data)),
    }
    print(pd.DataFrame(results).T.to_string())
