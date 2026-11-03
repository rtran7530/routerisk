"""Travel-time and delay models with strictly chronological evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["pickup_zone", "dropoff_zone"]
NUMERIC = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
FEATURES = CATEGORICAL + NUMERIC


def preprocessing():
    return ColumnTransformer([
        ("zones", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("calendar", StandardScaler(), NUMERIC),
    ])


def make_duration_model():
    return Pipeline([("features", preprocessing()), ("ridge", Ridge(alpha=30.0))])


def make_delay_model():
    return Pipeline([("features", preprocessing()),
                     ("logistic", LogisticRegression(C=0.2, max_iter=400))])


def build_features(pickup_zone, dropoff_zone, hour, weekday):
    if not 0 <= hour < 24 or not 0 <= weekday < 7:
        raise ValueError("Hour must be 0-23 and weekday must be 0-6")
    return pd.DataFrame([{
        "pickup_zone": str(pickup_zone), "dropoff_zone": str(dropoff_zone),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
    }])


def predict_duration(model, frame):
    return np.maximum(0, np.expm1(model.predict(frame[FEATURES])))


def conformal_radius(y_true, predicted, alpha=0.1):
    """Absolute residual split conformal quantile; infinity if calibration too small."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    scores = np.sort(np.abs(np.asarray(y_true) - np.asarray(predicted)))
    n = len(scores)
    if not n:
        raise ValueError("No calibration observations")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[rank - 1]) if rank <= n else float("inf")


def calibration_error(y, p, bins=10):
    """Expected absolute calibration gap across equally spaced probability bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(indices == j) * abs(y[indices == j].mean() - p[indices == j].mean())
                     for j in range(bins) if np.any(indices == j)))


def split_chronological(df):
    x = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    if len(x) < 100:
        raise ValueError("At least 100 observations are required")
    a, b = int(len(x) * .6), int(len(x) * .8)
    return x.iloc[:a].copy(), x.iloc[a:b].copy(), x.iloc[b:].copy()


def fit_models(train, calibration, threshold=30.0, alpha=.10):
    if threshold <= 0:
        raise ValueError("Delay threshold must be positive")
    duration = make_duration_model()
    duration.fit(train[FEATURES], np.log1p(train.duration_min))
    radius = conformal_radius(calibration.duration_min, predict_duration(duration, calibration), alpha)
    labels = (train.duration_min > threshold).astype(int)
    if labels.nunique() != 2:
        raise ValueError("Delay classification requires both threshold classes in training data")
    delay = make_delay_model()
    delay.fit(train[FEATURES], labels)
    raw_cal = delay.predict_proba(calibration[FEATURES])[:, 1]
    cal_labels = (calibration.duration_min > threshold).astype(int).to_numpy()
    # Calibration is performed only on the pre-test window.
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    isotonic.fit(raw_cal, cal_labels)
    return {"duration": duration, "delay": delay, "isotonic": isotonic,
            "radius": radius, "threshold": float(threshold), "alpha": float(alpha)}


def predict_delay(trained, frame):
    raw = trained["delay"].predict_proba(frame[FEATURES])[:, 1]
    return np.clip(trained["isotonic"].predict(raw), 0, 1)


def evaluate(trained, test, baseline_duration):
    actual = test.duration_min.to_numpy()
    prediction = predict_duration(trained["duration"], test)
    radius = trained["radius"]
    lower = np.maximum(0, prediction - radius)
    upper = prediction + radius
    p = predict_delay(trained, test)
    y = (actual > trained["threshold"]).astype(int)
    baseline_p = np.full(len(y), baseline_duration["delay_rate"])
    result = {
        "rows": int(len(test)),
        "mae_min": float(mean_absolute_error(actual, prediction)),
        "rmse_min": float(np.sqrt(mean_squared_error(actual, prediction))),
        "median_baseline_mae_min": float(mean_absolute_error(actual, np.full(len(y), baseline_duration["median"]))),
        "nominal_coverage": 1 - trained["alpha"],
        "empirical_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width_min": float(np.mean(upper - lower)),
        "delay_threshold_min": trained["threshold"],
        "observed_delay_rate": float(np.mean(y)),
        "predicted_delay_rate": float(np.mean(p)),
        "delay_brier": float(brier_score_loss(y, p)),
        "delay_baseline_brier": float(brier_score_loss(y, baseline_p)),
        "delay_calibration_error": calibration_error(y, p),
        "delay_roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }
    predictions = test[["tpep_pickup_datetime", "pickup_zone", "dropoff_zone", "duration_min"]].copy()
    predictions["predicted_min"] = prediction
    predictions["lower_min"] = lower
    predictions["upper_min"] = upper
    predictions["delay_probability"] = p
    predictions["delayed"] = y
    predictions["interval_covered"] = ((actual >= lower) & (actual <= upper)).astype(int)
    return result, predictions


def train_and_evaluate(df, alpha=.10, threshold=30.0):
    train, cal, test = split_chronological(df)
    trained = fit_models(train, cal, threshold=threshold, alpha=alpha)
    baseline = {"median": float(train.duration_min.median()),
                "delay_rate": float(np.mean(train.duration_min > threshold))}
    scores, predictions = evaluate(trained, test, baseline)
    scores.update({"train_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test),
                   "conformal_radius_min": trained["radius"]})
    return trained, scores, predictions


def monthly_backtest(frames, threshold=30.0, alpha=.10):
    """Initial month for training; next month calibration; later months untouched tests.

    One frozen model evaluates every subsequent month, making drift visible.
    """
    months = sorted(frames)
    if len(months) < 3:
        raise ValueError("Monthly backtest requires at least three distinct months")
    train, calibration = frames[months[0]], frames[months[1]]
    trained = fit_models(train, calibration, threshold=threshold, alpha=alpha)
    baseline = {"median": float(train.duration_min.median()),
                "delay_rate": float(np.mean(train.duration_min > threshold))}
    summaries, predictions = [], []
    for month in months[2:]:
        score, pred = evaluate(trained, frames[month], baseline)
        score["month"] = month
        pred["month"] = month
        summaries.append(score)
        predictions.append(pred)
    return trained, pd.DataFrame(summaries), pd.concat(predictions, ignore_index=True)
