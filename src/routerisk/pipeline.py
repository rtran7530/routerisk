"""Command line entry: python -m routerisk.pipeline --latest --months 4"""
import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import requests
from .clean import clean_trips
from .data import SOURCE, ZONES, download, latest_available_month, load_trips
from .database import route_summary, save_database
from .model import monthly_backtest, train_and_evaluate

ROOT = Path(__file__).resolve().parents[2]


def previous_months(year, month, count):
    start = year * 12 + month - 1
    return [divmod(start - i, 12) for i in range(count - 1, -1, -1)]


def run(paths, max_rows=50000, root=ROOT, threshold=30.0):
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    paths = [Path(p) for p in paths]
    if len(paths) not in (1,) and len(paths) < 3:
        raise ValueError("Supply one month, or at least three for monthly backtesting")
    frames, reports = {}, {}
    for path in paths:
        raw = load_trips(path, max_rows=max_rows)
        clean, quality = clean_trips(raw)
        key = str(clean.tpep_pickup_datetime.dt.to_period("M").mode()[0])
        clean = clean[clean.tpep_pickup_datetime.dt.to_period("M").astype(str) == key].copy()
        if len(clean) < 100:
            raise ValueError(f"Too few observations in month {key}")
        if key in frames:
            raise ValueError(f"Two files contain the same month: {key}")
        frames[key] = clean
        reports[key] = quality
    processed = root / "data" / "processed"
    artifacts = root / "artifacts"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
    combined.to_csv(processed / "cleaned_trips.csv", index=False)
    zones_path = root / "data" / "raw" / "taxi_zone_lookup.csv"
    zones = None
    if zones_path.exists():
        zones = pd.read_csv(zones_path)[["LocationID", "Borough", "Zone"]].rename(
            columns={"LocationID": "location_id", "Borough": "borough", "Zone": "zone"})
        zones["location_id"] = zones.location_id.astype(str)
    db_path = processed / "trips.sqlite"
    save_database(combined, db_path, zones)
    route_summary(db_path).to_csv(processed / "top_routes.csv", index=False)
    if len(frames) == 1:
        trained, score, preds = train_and_evaluate(combined, threshold=threshold)
        rows = [{"month": list(frames)[0], **score}]
        mode = "within_month_chronological_60_20_20"
    else:
        trained, month_scores, preds = monthly_backtest(frames, threshold=threshold)
        rows = month_scores.to_dict(orient="records")
        mode = "frozen_model_train_first_month_calibrate_second_test_later_months"
    preds.to_csv(processed / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(processed / "monthly_results.csv", index=False)
    joblib.dump(trained, artifacts / "model.joblib")
    report = {"mode": mode, "processed_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_files": [p.name for p in paths], "quality": reports,
              "test_results": rows,
              "limitations": "Month samples are random within each month; outcomes are taxi-zone trip durations, not street-level ETA. Conditional conformal coverage is not promised."}
    (artifacts / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description="RouteRisk: travel-time reliability and delay risk")
    parser.add_argument("--download", action="store_true", help="Download official TLC monthly parquet files")
    parser.add_argument("--latest", action="store_true", help="Locate latest published month")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--months", type=int, default=4, help="Number of consecutive months (1 or >=3)")
    parser.add_argument("--input", type=Path, nargs="+", help="Existing monthly TLC-format parquet/CSV files")
    parser.add_argument("--max-rows", type=int, default=50000, help="Sampled rows per month")
    parser.add_argument("--delay-threshold", type=float, default=30.0, help="Delay event: duration exceeds this number of minutes")
    args = parser.parse_args()
    if args.download or args.latest:
        if args.year is not None or args.month is not None:
            if args.year is None or args.month is None or not 1 <= args.month <= 12:
                parser.error("Specify valid --year and --month together")
            year, month = args.year, args.month
        else:
            year, month = latest_available_month()
        if args.months not in (1,) and args.months < 3:
            parser.error("--months must be 1 or at least 3")
        paths = []
        for y, m in previous_months(year, month, args.months):
            y, m = y, m + 1
            path = ROOT / "data" / "raw" / f"yellow_tripdata_{y}-{m:02d}.parquet"
            print(f"Using {y}-{m:02d}")
            paths.append(download(SOURCE.format(year=y, month=m), path))
        try:
            download(ZONES, ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
        except requests.RequestException:
            print("Zone lookup unavailable; using zone identifiers")
    elif args.input:
        paths = args.input
    else:
        parser.error("Provide --download --latest or --input FILE [FILE ...]")
    print(json.dumps(run(paths, max_rows=args.max_rows, threshold=args.delay_threshold), indent=2))


if __name__ == "__main__":
    main()
