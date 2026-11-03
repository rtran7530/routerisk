"""Create publication-style figures from real held-out RouteRisk outputs.

Run after routerisk.pipeline. No example or synthetic results are bundled.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def create_figures(root=ROOT):
    root = Path(root)
    monthly_file = root / "data" / "processed" / "monthly_results.csv"
    predictions_file = root / "data" / "processed" / "test_predictions.csv"
    if not monthly_file.exists() or not predictions_file.exists():
        raise FileNotFoundError("Run python -m routerisk.pipeline first to generate held-out results")
    monthly = pd.read_csv(monthly_file)
    predictions = pd.read_csv(predictions_file)
    output = root / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    if len(monthly) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.plot(monthly["month"].astype(str), monthly["empirical_coverage"], marker="o", linewidth=2, label="Observed")
        ax.axhline(float(monthly["nominal_coverage"].iloc[0]), linestyle="--", linewidth=1.5, label="Nominal target")
        ax.set(title="Prediction interval coverage over time", xlabel="Test month", ylabel="Fraction of trips covered", ylim=(0, 1))
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(output / "monthly_coverage.png", dpi=170)
        plt.close(fig)

    bins = np.linspace(0, 1, 11)
    grouped = predictions.copy()
    grouped["probability_bin"] = pd.cut(grouped["delay_probability"], bins=bins, include_lowest=True)
    calibration = grouped.groupby("probability_bin", observed=True).agg(
        predicted=("delay_probability", "mean"),
        observed=("delayed", "mean"),
        count=("delayed", "size")
    ).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(calibration["predicted"], calibration["observed"], s=np.clip(calibration["count"] / 5, 20, 180), alpha=.8, label="Held-out trips")
    ax.set(title="Delay probability calibration", xlabel="Mean predicted probability", ylabel="Observed exceedance rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "delay_calibration.png", dpi=170)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate RouteRisk figures from held-out trip data")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(f"Figures saved to {create_figures(args.root)}")


if __name__ == "__main__":
    main()
