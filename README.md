# RouteRisk

![Yellow taxis at night in Manhattan](https://commons.wikimedia.org/wiki/Special:Redirect/file/Taxi-cabs-New-York-0986.jpg)

<sub>Photo: [David R. Tribble](https://commons.wikimedia.org/wiki/File:Taxi-cabs-New-York-0986.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), unmodified.</sub>

RouteRisk, a statistical travel-time modeling project, uses historical [New York City yellow-taxi records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) to estimate trip durations and the probability of exceeding a specified time limit. The models are tested on data from later months to evaluate their predictive accuracy and reliability over time.

## Features

- Travel-time estimation with Ridge regression and a median-duration baseline
- Split-conformal prediction intervals with empirical coverage evaluation
- Delay probability estimates with isotonic calibration
- Monthly out-of-sample evaluation
- SQLite analysis and an interactive Streamlit dashboard

## Data

The analysis uses pickup and destination taxi zones together with the departure hour and weekday. Records with invalid timestamps, missing zones, duplicates, or implausible trip durations are excluded.

Only information available at departure is used for prediction. Drop-off timestamps determine observed duration, while recorded trip distance and fares are excluded as predictors. Taxi zones represent geographic areas rather than individual street routes.

## Mathematical Overview

Let $T>0$ be trip duration in minutes and $X=(Z_o,Z_d,H,W)\in\mathcal X$ contain departure information. The zone identifiers $Z_o,Z_d\in\mathcal Z$ denote pickup and destination zones, while $H\in\{0,\ldots,23\}$ and $W\in\{0,\ldots,6\}$ denote hour and weekday, respectively.

### Travel-Time Estimation

Ridge regression models $\log(1+T)$ using encoded zone and calendar features $\phi(x)\in\mathbb R^p$. The fitted travel-time estimate is

$$
\widehat{T}(x) = \max\bigl(0, \exp(\widehat{\beta}_0 + \widehat{\beta}^{\top}\phi(x)) - 1\bigr), \qquad x \in \mathcal{X}.
$$

### Prediction Intervals

Split-conformal prediction uses absolute calibration residuals $r_i=|T_i-\widehat T(X_i)|$. Their calibrated quantile $q_\alpha$ determines an interval for a new trip:

$$
C_\alpha(x)=\left[\max\{0,\widehat T(x)-q_\alpha\},\ \widehat T(x)+q_\alpha\right],\qquad \alpha\in(0,1).
$$

The default $\alpha=0.10$ corresponds to 90% nominal marginal coverage under exchangeability. Coverage is checked on later months, where travel patterns may differ.

### Delay Risk

For a travel-time limit $\tau>0$, the calibrated classifier estimates the exceedance probability

$$
p_\tau(x)=\Pr(T>\tau\mid X=x),\qquad x\in\mathcal X.
$$

The default threshold is 30 minutes. This is a duration threshold, not a scheduled-arrival deadline.

## Statistical Analysis

The first month is used for training and the second for calibration. Subsequent months provide out-of-sample observations for evaluating duration error and predictive reliability. Mean absolute error (MAE) is compared with a median-duration baseline.

### Interval Coverage

Empirical coverage is the proportion of test trips whose observed durations fall inside their prediction intervals:

$$
\widehat{\mathrm{Coverage}}=\frac{1}{n}\sum_{i=1}^{n}\mathbf 1\{T_i\in C_\alpha(X_i)\}.
$$

Here, $n\in\mathbb N$ is the number of test trips. The observed coverage is compared with the nominal 90% level.

### Delay Probability Accuracy

The Brier score compares predicted delay probabilities $\widehat p_i\in[0,1]$ with observed exceedance outcomes:

$$
\mathrm{BS}=\frac{1}{n}\sum_{i=1}^{n}\left(\widehat p_i-\mathbf 1\{T_i>\tau\}\right)^2.
$$

A lower score indicates better probabilistic accuracy. Calibration is assessed separately by comparing predicted probabilities with observed delay frequencies.

## Dashboard

The Streamlit dashboard provides a trip estimator alongside interactive views of prediction errors and monthly reliability. It also displays delay probability calibration, with year and month filters for selecting available evaluation periods. Light and dark themes are supported.

## Installation

Python 3.10 or newer is required. Run the appropriate commands from the repository directory.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

**Windows Command Prompt**

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, use `python` instead of `python3` in the commands below. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly.

## Run

Download the latest twelve available months and evaluate the models:

```bash
python3 -m routerisk.pipeline --download --latest --months 12 --max-rows 50000
```

Launch the dashboard:

```bash
python3 -m streamlit run app/dashboard.py
```

Run the automated tests:

```bash
python3 -m pytest -q
```

Results are written to `artifacts/` and `data/processed/`. Raw trip records are not committed, and `--max-rows` limits the number of processed records rather than download size.

## Limitations

RouteRisk estimates travel times between taxi zones rather than along specific street routes, and it does not incorporate live traffic conditions. Trips shorter than 2 minutes or longer than 180 minutes are excluded; therefore, the results apply only to the retained records, not to all NYC taxi trips.

Because the models use historical observations, changing travel conditions may reduce predictive accuracy and calibration. Similarly, conformal coverage guarantees require exchangeability, which may not hold when evaluating later months.

## References

- Angelopoulos, A. N., and Bates, S. (2023), [*Conformal Prediction: A Gentle Introduction*](https://doi.org/10.1561/2200000101)
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [NYC Taxi & Limousine Commission: Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## License

MIT License. See [LICENSE](LICENSE). The photograph is separately licensed under CC BY-SA 3.0, and TLC data retain their own terms.
