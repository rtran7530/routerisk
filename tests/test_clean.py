import pandas as pd
from routerisk.clean import clean_trips


def test_rejects_invalid_and_duplicates():
    rows = []
    for i in range(100):
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=10*i)
        rows.append({"tpep_pickup_datetime":start, "tpep_dropoff_datetime":start+pd.Timedelta(minutes=12),
                     "PULocationID": 100, "DOLocationID": 120, "trip_distance":2.5})
    rows.append(rows[0].copy())
    rows.append({**rows[0], "tpep_pickup_datetime":pd.Timestamp("2025-01-03"), "tpep_dropoff_datetime":pd.Timestamp("2025-01-02")})
    clean, quality = clean_trips(pd.DataFrame(rows))
    assert len(clean) == 100
    assert quality["duplicate_rows_removed"] == 1
    assert quality["rows_removed"] == 2
    assert clean.duration_min.min() == 12
