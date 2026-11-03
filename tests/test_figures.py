import numpy as np
import pandas as pd
from routerisk.figures import create_figures


def test_create_figures(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"month": ["2025-03", "2025-04"],
                  "empirical_coverage": [.87, .89],
                  "nominal_coverage": [.9, .9]}).to_csv(processed / "monthly_results.csv", index=False)
    pd.DataFrame({"delay_probability": np.linspace(.05, .95, 30),
                  "delayed": np.array([0, 1] * 15)}).to_csv(processed / "test_predictions.csv", index=False)
    output = create_figures(tmp_path)
    assert (output / "monthly_coverage.png").stat().st_size > 0
    assert (output / "delay_calibration.png").stat().st_size > 0
