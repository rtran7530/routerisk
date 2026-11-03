"""Deterministic cleaning rules, with an auditable quality report."""
import numpy as np
import pandas as pd
from .data import COLUMNS


def clean_trips(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")
    x = df[COLUMNS].copy()
    report = {"input_rows": len(x)}
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        x[col] = pd.to_datetime(x[col], errors="coerce")
    for col in ["PULocationID", "DOLocationID", "trip_distance"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    report["missing_values_by_column"] = {k: int(v) for k, v in x.isna().sum().items()}
    x = x.dropna(subset=COLUMNS)
    before = len(x)
    x = x.drop_duplicates()
    report["duplicate_rows_removed"] = before - len(x)
    x["duration_min"] = (x.tpep_dropoff_datetime - x.tpep_pickup_datetime).dt.total_seconds() / 60
    # Exclude impossible/very atypical cases for this *scoped* modeling exercise.
    valid = (
        x.duration_min.between(2, 180)
        & x.trip_distance.between(0.2, 100)
        & x.PULocationID.between(1, 263)
        & x.DOLocationID.between(1, 263)
    )
    x = x.loc[valid].copy()
    x["hour"] = x.tpep_pickup_datetime.dt.hour
    x["weekday"] = x.tpep_pickup_datetime.dt.dayofweek
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24)
    x["weekday_sin"] = np.sin(2 * np.pi * x.weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * x.weekday / 7)
    x["pickup_zone"] = x.PULocationID.astype(int).astype(str)
    x["dropoff_zone"] = x.DOLocationID.astype(int).astype(str)
    x = x.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    report["output_rows"] = len(x)
    report["rows_removed"] = report["input_rows"] - len(x)
    report["retention_rate"] = round(len(x) / max(report["input_rows"], 1), 5)
    if len(x) < 60:
        raise ValueError("Too few valid trips after cleaning (need at least 60)")
    return x, report
