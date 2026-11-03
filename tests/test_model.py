import numpy as np
import pandas as pd
from routerisk.model import (build_features, calibration_error, conformal_radius,
                              monthly_backtest, train_and_evaluate)


def fixture(month, n=240, seed=2):
    rng = np.random.default_rng(seed + month)
    time = pd.date_range(f"2025-{month:02d}-01", periods=n, freq="1h")
    minutes = np.clip(24 + 5 * np.sin(np.arange(n) / 12) + rng.normal(0, 10, n), 2, 85)
    hour, weekday = time.hour, time.dayofweek
    return pd.DataFrame({
        "tpep_pickup_datetime": time,
        "pickup_zone": [str(100 + i % 3) for i in range(n)],
        "dropoff_zone": [str(120 + i % 4) for i in range(n)],
        "duration_min": minutes,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    })


def test_conformal_order_statistic():
    assert conformal_radius(np.arange(10.), np.zeros(10), alpha=.2) == 8
    assert np.isinf(conformal_radius([1., 2.], [1., 1.], alpha=.1))


def test_features():
    x = build_features("100", "120", 9, 2)
    assert x.iloc[0].pickup_zone == "100"
    assert np.isclose(x.iloc[0].hour_sin, np.sin(2 * np.pi * 9 / 24))


def test_single_month():
    trained, scores, preds = train_and_evaluate(fixture(1))
    assert len(preds) == 48
    assert 0 <= scores["empirical_coverage"] <= 1
    assert 0 <= scores["delay_brier"] <= 1
    assert 0 <= preds.delay_probability.min() <= preds.delay_probability.max() <= 1
    assert trained["threshold"] == 30


def test_frozen_monthly_backtest():
    trained, results, preds = monthly_backtest({"2025-01": fixture(1), "2025-02": fixture(2),
                                                "2025-03": fixture(3), "2025-04": fixture(4)})
    assert results.month.to_list() == ["2025-03", "2025-04"]
    assert len(preds) == 480
    assert trained["radius"] >= 0


def test_calibration_error():
    assert calibration_error([0, 1], [.0, 1.]) == 0
