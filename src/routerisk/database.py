"""SQLite layer for reproducible SQL exploratory analysis."""
import sqlite3
from pathlib import Path
import pandas as pd


def save_database(df: pd.DataFrame, path: Path, zones: pd.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df[["tpep_pickup_datetime", "pickup_zone", "dropoff_zone", "duration_min", "hour", "weekday"]].copy()
    records["tpep_pickup_datetime"] = records.tpep_pickup_datetime.astype(str)
    with sqlite3.connect(path) as conn:
        records.to_sql("trips", conn, if_exists="replace", index=False)
        if zones is not None:
            zones.to_sql("zones", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_route ON trips(pickup_zone, dropoff_zone)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hour ON trips(hour)")


def route_summary(path: Path, min_trips: int = 20) -> pd.DataFrame:
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query("""
            SELECT pickup_zone, dropoff_zone,
                   COUNT(*) AS trips,
                   ROUND(AVG(duration_min), 2) AS mean_duration_min,
                   ROUND(AVG(duration_min * duration_min) - AVG(duration_min) * AVG(duration_min), 2) AS variance_min2
            FROM trips GROUP BY pickup_zone, dropoff_zone
            HAVING COUNT(*) >= ? ORDER BY trips DESC LIMIT 30
        """, conn, params=(min_trips,))
