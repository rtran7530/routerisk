"""RouteRisk: historical travel-time estimation and reliability analysis."""
from __future__ import annotations

from datetime import datetime, timezone
from calendar import month_abbr
import json
from html import escape
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import declare_component

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from routerisk.model import build_features, predict_delay, predict_duration  # noqa: E402

st.set_page_config(
    page_title="RouteRisk", page_icon="🚕", layout="wide", initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

THEMES = {
    "light": dict(bg="#f8fafc", sidebar="#f0f4f8", card="#ffffff", text="#172638",
                  muted="#465a70", border="#ccd7e2", accent="#245e82", accent_text="#ffffff",
                  grid="#e0e8f0", soft="#e3edf5", field="#ffffff"),
    "dark": dict(bg="#101820", sidebar="#17232e", card="#1d2c39", text="#f2f6fa",
                 muted="#c0cfda", border="#475a6a", accent="#8dc8e6", accent_text="#102230",
                 grid="#324554", soft="#263b4a", field="#243542"),
}
if "appearance" not in st.session_state:
    st.session_state.appearance = "light"
P = THEMES[st.session_state.appearance]
HIGHLIGHT = "#f4cd72" if st.session_state.appearance == "dark" else P["text"]

# All CSS values are locally defined constants, never user input.
st.markdown(f"""
<style>

:root {{
  color-scheme: {st.session_state.appearance};
  --text-color: {P['text']};
  --background-color: {P['bg']};
  --secondary-background-color: {P['sidebar']};
  --primary-color: {P['accent']};
  --border-color: {P['border']};
  --routerisk-text: {P['text']};
  --routerisk-muted: {P['muted']};
}}
html, body, .stApp, [data-testid='stAppViewContainer'], [data-testid='stMain'] {{
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif !important; background:{P['bg']} !important; color:{P['text']} !important;
}}
[data-testid='stSidebar'], [data-testid='stSidebarContent'] {{
  background:{P['sidebar']} !important; color:{P['text']} !important; border-color:{P['border']} !important;
}}
[data-testid='stHeader'], [data-testid='stToolbar'], [data-testid='stDecoration'],
[data-testid='stStatusWidget'], [data-testid='stAppDeployButton'],
[data-testid='stMainMenu'], [data-testid='stHeaderActionElements'],
#MainMenu, footer {{ display:none !important; visibility:hidden !important; }}
.block-container {{ max-width:1420px; padding-top:1.6rem; padding-bottom:1.8rem; }}
h1,h2,h3,h4,h5,h6, .stMarkdown, [data-testid='stMarkdownContainer'],
[data-testid='stWidgetLabel'], [data-testid='stWidgetLabel'] *, [data-testid='stText'] {{ color:{P['text']} !important; }}
.stApp, .stApp p, .stApp label, .stApp small {{ color:{P['text']} !important; }}
[data-testid='stSidebar'] p, [data-testid='stSidebar'] label {{ color:{P['text']} !important; }}
h1 {{ font-size:2rem !important; letter-spacing:-.025em; font-weight:700 !important; }}
h2 {{ font-size:1.25rem !important; }}
h3 {{ font-size:1.06rem !important; }}
[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] * {{ color:{P['muted']} !important; }}
[data-testid='stMetric'] {{ border:1px solid {P['border']}; background:{P['card']}; padding:15px 17px; border-radius:9px; }}
[data-testid='stMetricLabel'], [data-testid='stMetricLabel'] * {{ color:{P['muted']} !important; }}
[data-testid='stMetricValue'], [data-testid='stMetricValue'] * {{ color:{P['text']} !important; font-size:1.55rem !important; }}
[data-testid='stMetricLabel'] svg {{ display:none !important; }}
[data-baseweb='select'] > div, [data-baseweb='input'] > div,
[data-testid='stNumberInput'] input, [data-baseweb='base-input'] input,
[data-testid='stTextInput'] input {{ background:{P['field']} !important; color:{P['text']} !important;
  border-color:{P['border']} !important; }}
[data-baseweb='select'] input, [data-baseweb='select'] span,
[data-baseweb='select'] div, [data-baseweb='input'] input,
[data-testid='stSelectbox'] div, [data-testid='stMultiSelect'] div {{ color:{P['text']} !important; }}
[data-baseweb='select'] svg, [data-testid='stSelectbox'] svg {{ fill:{P['muted']} !important; color:{P['muted']} !important; }}
[data-baseweb='tag'] {{ background:{P['soft']} !important; }}
[data-baseweb='tag'] > span, [data-baseweb='tag'] [role='presentation'] {{ color:{P['text']} !important; }}
[data-baseweb='select'] [aria-selected='true'] {{ background:{P['soft']} !important; }}
[data-baseweb='select'] [data-baseweb='input'] {{ background:{P['field']} !important; }}
[data-testid='stSlider'] p, [data-testid='stSlider'] span {{ color:{P['muted']} !important; }}
[data-baseweb='tag'] span {{ color:{P['text']} !important; }}
[data-baseweb='tag'] svg {{ fill:{P['text']} !important; }}
[data-baseweb='popover'], [data-baseweb='popover'] > div,
[role='listbox'], [role='option'], [data-baseweb='menu'], [data-baseweb='menu'] li {{
  background:{P['card']} !important; color:{P['text']} !important;
}}
[role='option'] *, [data-baseweb='menu'] li * {{ color:{P['text']} !important; }}
[role='option']:hover, [data-baseweb='menu'] li:hover {{ background:{P['soft']} !important; }}
[data-testid='stTabs'] [role='tab'] {{ color:{P['muted']} !important; }}
[data-testid='stTabs'] [role='tab'][aria-selected='true'] {{ color:{P['text']} !important; border-color:{P['accent']} !important; }}
[data-testid='stTabs'] [role='tab']:hover {{ color:{P['text']} !important; }}
[data-testid='stTabs'] [data-baseweb='tab-highlight'] {{ background:{P['accent']} !important; }}
[data-testid='stTabs'] [data-baseweb='tab-border'] {{ background:{P['border']} !important; }}
[data-testid='stSlider'] [role='slider'] {{ background:{P['accent']} !important; }}
.stButton > button, .stDownloadButton > button {{ border:1px solid {P['border']} !important;
  background:{P['card']} !important; color:{P['text']} !important; border-radius:8px; }}
.stButton > button *, .stDownloadButton > button * {{ color:{P['text']} !important; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:{P['soft']} !important; border-color:{P['accent']} !important; }}
[data-testid='stDataFrame'], [data-testid='stExpander'], hr {{ border-color:{P['border']} !important; }}
.st-key-theme_toggle button {{ border-radius:50% !important; width:42px !important;
  min-width:42px !important; height:42px !important; padding:0 !important;
  background:{P['card']} !important; color:{P['text']} !important;
  border:1px solid {P['border']} !important; }}
[data-testid='stAlert'] {{ background:{P['soft']} !important; color:{P['text']} !important; }}
[data-testid='stAlert'] * {{ color:{P['text']} !important; }}
[data-testid='stDataFrame'] {{ color:{P['text']} !important; }}
/* RouteRisk cards use ordinary HTML, not native metric tooltips. */
.rr-metric {{ border:1px solid {P['border']}; background:{P['card']};
  padding:16px 17px; border-radius:9px; min-height:104px; }}
.rr-metric-title {{ font-size:.85rem; font-weight:600; color:{P['muted']} !important;
  display:flex; align-items:center; gap:6px; }}
.rr-metric-number {{ font-size:1.7rem; font-weight:650; color:{HIGHLIGHT} !important;
  font-variant-numeric:tabular-nums; margin-top:5px; line-height:1.25; }}
.rr-indicator {{ display:inline-block; font-weight:800; font-size:1.07rem;
  color:{P['accent']} !important; }}
.rr-notes {{ margin-top:9px; font-size:.83rem; color:{P['muted']} !important; }}
.rr-build {{ color:{P['muted']} !important; font-size:.72rem; text-align:right; }}
/* Consistent, accessible table contrast in both themes. */
.rr-table-wrap {{ margin-top:12px; max-height:420px; overflow:auto;
  border:1px solid {P['border']}; border-radius:9px; background:{P['card']}; }}
.rr-table {{ width:100%; border-collapse:separate; border-spacing:0;
  font-size:.79rem; font-variant-numeric:tabular-nums; color:{P['text']} !important; }}
.rr-table thead th {{ position:sticky; top:0; background:{P['soft']} !important;
  color:{P['text']} !important; font-weight:700; text-align:left;
  padding:12px 11px; border-bottom:2px solid {P['border']}; white-space:nowrap; z-index:1; }}
.rr-table tbody td {{ color:{P['text']} !important; background:{P['card']} !important;
  padding:9px 11px; border-bottom:1px solid {P['border']}; white-space:nowrap; }}
.rr-table tbody tr:hover td {{ background:{P['soft']} !important; }}
.rr-table-note {{ color:{P['muted']} !important; font-size:.8rem; margin-top:6px; }}
[data-testid='stRadio'] [role='radiogroup'] {{ gap:20px; flex-wrap:wrap; }}
[data-testid='stRadio'] [role='radiogroup'] label {{ cursor:pointer; }}
[data-testid='stRadio'] [role='radiogroup'] label p {{ font-weight:600; }}
[data-testid='stTooltipIcon'], [data-testid='stTooltipHoverTarget'] {{ display:none !important; }}

/* Keep link text visible in both modes. */
a, a:visited {{ color:{P['accent']} !important; }}
a:hover {{ text-decoration:underline !important; }}
</style>
""", unsafe_allow_html=True)

live_component = declare_component("routerisk_live_slider", path=str(ROOT / "app" / "components" / "live_slider"))


def live_slider(label: str, value: int, minimum: int, maximum: int, key: str, unit: str = "") -> int:
    result = live_component(label=label, value=int(value), min=minimum, max=maximum,
                            step=1, unit=unit, ink=P["text"], muted=P["muted"],
                            accent=P["accent"], bg=P["sidebar"] if key.startswith("duration") else P["bg"], key=key, default=int(value))
    return int(np.clip(value if result is None else result, minimum, maximum))


def make_figure(height=320):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=16, r=12, t=20, b=36),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif", color=P["text"], size=12),
                      template="none", hoverlabel=dict(bgcolor=P["card"], font_color=P["text"], bordercolor=P["border"]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, itemwidth=85,
                                  font=dict(color=P["text"], size=12), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["border"],
                     tickfont=dict(color=P["muted"]), title_font=dict(color=P["text"]))
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["border"],
                     tickfont=dict(color=P["muted"]), title_font=dict(color=P["text"]))
    return fig


@st.cache_data(show_spinner=False)
def read_json(path, modified):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def read_predictions(path, modified):
    df = pd.read_csv(path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    if "month" not in df:
        df["month"] = df["tpep_pickup_datetime"].dt.to_period("M").astype(str)
    df["error_min"] = df["predicted_min"] - df["duration_min"]
    return df


@st.cache_data(show_spinner=False)
def read_zone_labels(path, modified):
    source = Path(path)
    if source.exists():
        records = pd.read_csv(source)
        return {f"{r.Zone} · {r.Borough} ({r.LocationID})": str(r.LocationID)
                for r in records.itertuples() if 1 <= int(r.LocationID) <= 263}
    return {f"Taxi zone {n}": str(n) for n in range(1, 264)}


@st.cache_resource(show_spinner=False)
def read_model(path, modified):
    return joblib.load(path)


def file_version(path):
    return path.stat().st_mtime_ns if path.exists() else 0


def metric_card(col, title, value, help_text="", direction=None):
    """Display a stable, readable metric without Streamlit hover popovers."""
    marker = {"lower": "↓", "higher": "↑", "target": "↔"}.get(direction, "")
    marker_html = f'<span class="rr-indicator" aria-label="{escape(direction or "")}">{marker}</span>' if marker else ""
    with col:
        st.markdown(
            f'<div class="rr-metric"><div class="rr-metric-title">{escape(title)} {marker_html}</div>'
            f'<div class="rr-metric-number">{escape(str(value))}</div></div>',
            unsafe_allow_html=True,
        )


TABLE_COLUMNS = {
    "month": "Month", "pickup_zone": "Pickup zone", "dropoff_zone": "Destination zone",
    "duration_min": "Observed (min)", "predicted_min": "Predicted (min)",
    "lower_min": "Lower (min)", "upper_min": "Upper (min)",
    "delay_probability": "Delay probability", "interval_covered": "Covered",
    "mae_min": "MAE (min)", "median_baseline_mae_min": "Baseline MAE (min)",
    "empirical_coverage": "Coverage", "delay_brier": "Brier score",
    "rows": "Trips", "mean_interval_width_min": "Interval width (min)",
}

def results_table(frame: pd.DataFrame, max_rows: int = 25):
    """Render a short, theme-aware read-only table; retain raw names in exported CSV."""
    shown = frame.head(max_rows).rename(columns=TABLE_COLUMNS).copy()
    for col in shown.select_dtypes(include=["float"]).columns:
        shown[col] = shown[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    table = shown.to_html(index=False, classes="rr-table", border=0, escape=True)
    st.markdown(f'<div class="rr-table-wrap">{table}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rr-table-note">Showing {len(shown):,} of {len(frame):,} observations.</div>', unsafe_allow_html=True)

header, toggle = st.columns([13, 1], vertical_alignment="center")
with header:
    st.title("RouteRisk")
with toggle:
    icon = "☾" if st.session_state.appearance == "light" else "☀"
    def toggle_appearance():
        st.session_state.appearance = "dark" if st.session_state.appearance == "light" else "light"
    st.button(icon, key="theme_toggle", on_click=toggle_appearance)

metrics_path = ROOT / "artifacts" / "metrics.json"
pred_path = ROOT / "data" / "processed" / "test_predictions.csv"
model_path = ROOT / "artifacts" / "model.joblib"
if not all(p.exists() for p in (metrics_path, pred_path, model_path)):
    st.info("No results yet. Run the command below, then refresh the dashboard.")
    st.code("python3 -m routerisk.pipeline --download --latest --months 4 --max-rows 50000", language="bash")
    st.stop()

report = read_json(str(metrics_path), file_version(metrics_path))
pred = read_predictions(str(pred_path), file_version(pred_path))
model = read_model(str(model_path), file_version(model_path))
months = sorted(pred["month"].dropna().astype(str).unique().tolist())
monthly = pd.DataFrame(report.get("test_results", []))
threshold = float(model["threshold"])
nominal = 1 - float(model["alpha"])

source_months = sorted([str(s).split("yellow_tripdata_")[-1].split(".parquet")[0]
                       for s in report.get("source_files", []) if "yellow_tripdata_" in s])
latest_period = source_months[-1] if source_months else (months[-1] if months else None)
try:
    through = pd.Period(latest_period, freq="M").strftime("%b %Y") if latest_period else "Unknown"
except ValueError:
    through = latest_period or "Unknown"
refreshed = report.get("processed_at_utc")
if refreshed:
    try:
        refreshed_display = datetime.fromisoformat(refreshed.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        refreshed_display = str(refreshed)
else:
    refreshed_display = "Not recorded"
training_note = ""
if len(source_months) >= 3:
    training_note = f"  ·  Training {source_months[0]}  ·  Calibration {source_months[1]}"
st.caption(f"Data through {through}  ·  Updated {refreshed_display}  ·  Delay threshold {threshold:g} min" + training_note)
st.markdown('<div class="rr-build">Dashboard revision 2026.11.03c</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Filters")
    available_years = sorted({int(m[:4]) for m in months})
    chosen_years = st.multiselect("Year", available_years, default=available_years,
                                  key="evaluation_years")
    available_month_numbers = sorted({int(m[5:7]) for m in months if int(m[:4]) in chosen_years})
    chosen_month_numbers = st.multiselect("Month", available_month_numbers,
                                          default=available_month_numbers,
                                          format_func=lambda n: month_abbr[n], key="evaluation_month_numbers",
                                           )
    chosen_months = [m for m in months if int(m[:4]) in chosen_years and int(m[5:7]) in chosen_month_numbers]
    st.caption(f"{len(chosen_months)} of {len(months)} evaluation months selected")
    min_duration = live_slider("Minimum duration", 0, 0, 180, "duration_min", " min")
    max_duration = live_slider("Maximum duration", 180, 0, 180, "duration_max", " min")
    if min_duration > max_duration:
        min_duration, max_duration = max_duration, min_duration
    zone_ids = sorted(pred["pickup_zone"].dropna().astype(str).unique(), key=lambda z: int(z) if z.isdigit() else 9999)
    pickup_filter = st.selectbox("Pickup taxi zone", ["All zones", *zone_ids])

selected = pred[pred["month"].isin(chosen_months) & pred["duration_min"].between(min_duration, max_duration)].copy()
if pickup_filter != "All zones":
    selected = selected[selected["pickup_zone"].astype(str) == pickup_filter]

view = st.radio("Analysis view", ["Overview", "Monthly reliability", "Calibration", "Trip estimator"],
                horizontal=True, label_visibility="collapsed", key="active_view")
if view == "Overview":
    st.subheader("Held-out trips")
    if selected.empty:
        st.warning("No trips match these filters.")
    else:
        actual = selected["duration_min"].to_numpy()
        forecast = selected["predicted_min"].to_numpy()
        cov = float(selected["interval_covered"].mean())
        brier = float(np.mean((selected["delay_probability"] - (selected["duration_min"] > threshold).astype(int)) ** 2))
        a,b,c,d = st.columns(4)
        metric_card(a, "MAE", f"{np.mean(np.abs(forecast-actual)):.1f} min", "Mean absolute error in minutes. Lower is better.", direction="lower")
        metric_card(b, "Interval coverage", f"{cov:.1%}", f"Share of trips inside the prediction interval. Target: {nominal:.0%}; values substantially above or below the target need interpretation.", direction="target")
        metric_card(c, "Brier score", f"{brier:.3f}", "Mean squared error of predicted delay probabilities. Lower is better (0 is perfect).", direction="lower")
        metric_card(d, "Test trips", f"{len(selected):,}", "Number of held-out trips matching the selected filters. No preferred direction.")
        left,right=st.columns(2)
        with left:
            st.markdown("#### Observed vs predicted")
            fig=make_figure()
            sample=selected.sample(min(1800,len(selected)),random_state=23)
            fig.add_trace(go.Scattergl(x=sample.duration_min,y=sample.predicted_min,mode="markers",name="Trips",
                                       marker=dict(color=P['accent'],size=5,opacity=.42)))
            fig.add_trace(go.Scatter(x=[0,180],y=[0,180],name="Exact agreement",mode="lines",
                                     line=dict(color=P['muted'],dash="longdash",width=3)))
            fig.update_xaxes(title="Observed (min)",range=[0,180])
            fig.update_yaxes(title="Predicted (min)",range=[0,180])
            st.plotly_chart(fig,use_container_width=True,key="accuracy_chart")
        with right:
            st.markdown("#### Prediction error")
            fig=make_figure()
            hist_values = selected.error_min.to_numpy(dtype=float)
            hist_values = hist_values[np.isfinite(hist_values)]
            counts, edges = np.histogram(hist_values, bins=45)
            fig.add_trace(go.Bar(x=(edges[:-1]+edges[1:])/2, y=counts,
                                 width=np.diff(edges), name="Trips", marker_color=P['accent']))
            fig.add_vline(x=0,line_color=P['muted'],line_dash="dash")
            fig.update_xaxes(title="Predicted − observed (min)")
            fig.update_yaxes(title="Trip count")
            st.plotly_chart(fig,use_container_width=True,key="error_chart")

if view == "Monthly reliability":
    st.subheader("Monthly reliability")
    shown_monthly = monthly[monthly["month"].astype(str).isin(chosen_months)].copy() if "month" in monthly else pd.DataFrame()
    if shown_monthly.empty:
        st.info("No monthly results for the selected years and months.")
    else:
        fig=make_figure(340)
        fig.add_trace(go.Scatter(x=shown_monthly.month,y=shown_monthly.empirical_coverage*100,mode="lines+markers",
                                 name="Empirical",line=dict(color=P['accent'],width=3)))
        fig.add_hline(y=nominal*100,line_color=P['muted'],line_dash="dash",annotation_text=f"Target {nominal:.0%}")
        fig.update_xaxes(title="Test month",type="category")
        fig.update_yaxes(title="Covered trips (%)",range=[0,100])
        st.plotly_chart(fig,use_container_width=True,key="monthly_coverage")
        fig=make_figure(320)
        fig.add_trace(go.Scatter(x=shown_monthly.month,y=shown_monthly.mae_min,mode="lines+markers",name="Ridge",line=dict(color=P['accent'],width=3)))
        fig.add_trace(go.Scatter(x=shown_monthly.month,y=shown_monthly.median_baseline_mae_min,mode="lines+markers",name="Median baseline",
                                 line=dict(color=P['muted'],width=2)))
        fig.update_xaxes(title="Test month",type="category")
        fig.update_yaxes(title="Mean absolute error (min)")
        st.plotly_chart(fig,use_container_width=True,key="monthly_mae")
        with st.expander("Monthly metrics"):
            results_table(shown_monthly)

if view == "Calibration":
    st.subheader("Delay probability calibration")
    bins=st.select_slider("Probability bins",options=[5,8,10,12,15,20],value=10)
    if selected.empty:
        st.warning("No trips match these filters.")
    else:
        groups=selected.copy()
        groups["group"]=pd.cut(groups.delay_probability,bins=np.linspace(0,1,bins+1),include_lowest=True)
        grouped=groups.groupby("group",observed=True).agg(predicted=("delay_probability","mean"),
                                                            observed=("delayed","mean"),count=("delayed","size"))
        fig=make_figure(420)
        fig.add_trace(go.Scatter(x=grouped.predicted,y=grouped.observed,mode="lines+markers",name="Observed",
                                 marker=dict(color=P['accent'],size=10),line=dict(color=P['accent']),customdata=grouped['count'],
                                 hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Trips %{customdata}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line=dict(color=P['muted'],dash="longdash",width=3)))
        fig.update_xaxes(title="Predicted probability",range=[0,1],tickformat=".0%")
        fig.update_yaxes(title="Observed delay frequency",range=[0,1],tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True,key="calibration_chart")
        st.metric(f"Observed trips over {threshold:g} min",f"{selected['delayed'].mean():.1%}")
        with st.expander("Observations"):
            fields=["month","pickup_zone","dropoff_zone","duration_min","predicted_min","lower_min","upper_min","delay_probability","interval_covered"]
            results_table(selected[fields], max_rows=25)
            st.download_button("Export CSV",selected.to_csv(index=False),"routerisk_evaluation.csv","text/csv")

if view == "Trip estimator":
    st.subheader("Trip estimate")
    zones_path=ROOT/"data"/"raw"/"taxi_zone_lookup.csv"
    labels=read_zone_labels(str(zones_path),file_version(zones_path))
    options=sorted(labels)
    c1,c2=st.columns(2)
    with c1:
        origin=st.selectbox("Pickup taxi zone",options,key="origin")
        weekday=st.selectbox("Departure day",list(range(7)),format_func=lambda d:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d])
    with c2:
        destination=st.selectbox("Destination taxi zone",options,index=min(1,len(options)-1),key="destination")
        hour=live_slider("Departure hour",9,0,23,"departure_hour",":00")
    features=build_features(labels[origin],labels[destination],hour,weekday)
    estimate=float(predict_duration(model["duration"],features)[0])
    prob=float(predict_delay(model,features)[0])
    radius=float(model["radius"])
    lower=max(0,estimate-radius)
    upper=estimate+radius
    a,b,c=st.columns(3)
    metric_card(a,"Predicted duration",f"{estimate:.1f} min","Estimated travel time for the selected departure conditions.")
    metric_card(b,f"{nominal:.0%} prediction interval",f"{lower:.1f}–{upper:.1f} min","Marginal interval calibrated on historical trips; not guaranteed for an individual zone pair.")
    metric_card(c,f"P(duration > {threshold:g} min)",f"{prob:.1%}","Estimated chance that the trip lasts longer than the threshold.")
    st.caption("Duration: predicted trip time   ·   Interval: historical marginal coverage   ·   Probability: chance of exceeding the threshold")
    fig=make_figure(165)
    fig.add_trace(go.Scatter(x=[lower,upper],y=[0,0],mode="lines",name="Prediction interval",line=dict(color=P['accent'],width=12)))
    fig.add_trace(go.Scatter(x=[estimate],y=[0],mode="markers",name="Estimate",
                             marker=dict(color="#f59e0b" if st.session_state.appearance == "dark" else "#b45309",
                                         size=19, symbol="diamond", line=dict(color=P['bg'], width=2))))
    fig.update_yaxes(visible=False,range=[-.25,.25])
    fig.update_xaxes(title="Travel time (min)",range=[max(0,lower-5),upper+5])
    st.plotly_chart(fig,use_container_width=True,key="estimate_chart")

st.markdown("Source: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  ·  Historical estimates, not live traffic")
