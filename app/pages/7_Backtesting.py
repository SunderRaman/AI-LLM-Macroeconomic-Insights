import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Paths
WB_CLEAN = os.path.join("data", "processed", "wb_macro_clean.csv")
FORECASTS_ALL = os.path.join("data", "processed", "forecasts", "forecasts_all.csv")
BT_SUMMARY = os.path.join("data", "processed", "backtesting", "backtest_metrics_by_horizon.csv")
BT_DETAILS = os.path.join("data", "processed", "backtesting", "backtest_details_multi.csv")


@st.cache_data
def load_data():
    hist = pd.read_csv(WB_CLEAN)
    fc = pd.read_csv(FORECASTS_ALL)
    bt_sum = pd.read_csv(BT_SUMMARY)
    bt_det = pd.read_csv(BT_DETAILS)
    # ensure types
    hist["year"] = hist["year"].astype(int)
    fc["year"] = fc["year"].astype(int)
    bt_sum["horizon"] = bt_sum["horizon"].astype(int)
    return hist, fc, bt_sum, bt_det


def get_available_options(hist, fc):
    countries = sorted(list(set(hist["country"].unique()) & set(fc["country"].unique())))
    indicators = sorted(list(set(hist.columns) - {"country", "country_code", "year"}))
    return countries, indicators


def compute_uncertainty_band(fc_sel, bt_sum_sel, last_hist_year):
    """
    For each forecast year, assign a % error from backtesting by horizon.
    horizon = year - last_hist_year
    For horizon > max_horizon_backtested, reuse the max horizon MAPE.
    """
    if bt_sum_sel.empty:
        # no backtest info => no band
        fc_sel = fc_sel.copy()
        fc_sel["mape_pct"] = 0.0
        fc_sel["lower"] = fc_sel["forecast_mean"]
        fc_sel["upper"] = fc_sel["forecast_mean"]
        return fc_sel

    # build dict: horizon -> mape_pct
    h2m = {int(row["horizon"]): float(row["mape_pct"]) for _, row in bt_sum_sel.iterrows()}
    max_h = max(h2m.keys())

    def pick_mape(year):
        h = year - last_hist_year
        if h <= 0:
            return 0.0
        h_use = h if h in h2m else max_h
        return h2m.get(h_use, 0.0)

    if "forecast_mean" in fc_sel.columns:
        fc_sel = fc_sel.rename(columns={"forecast_mean": "value"})

    fc = fc_sel.copy()
    #st.write("FC columns:", fc_sel.columns.tolist())
    #st.write("Unique indicators in forecasts:", fc["indicator"].unique())
    # st.write("Selected indicator:", indicator)
    fc["horizon"] = fc["year"].apply(lambda y: max(y - last_hist_year, 0))
    fc["mape_pct"] = fc["year"].apply(pick_mape)

    fc["lower"] = fc["value"] * (1 - fc["mape_pct"] / 100.0)
    fc["upper"] = fc["value"] * (1 + fc["mape_pct"] / 100.0)
    return fc


def main():
    st.title("Module D — Backtesting & Uncertainty")

    hist, fc, bt_sum, bt_det = load_data()
    countries, indicators = get_available_options(hist, fc)

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
    with col2:
        indicator = st.selectbox("Indicator", indicators, index=indicators.index("gdp_current_usd") if "gdp_current_usd" in indicators else 0)

    st.markdown(f"### {country} — {indicator}")

    # Filter backtest summary for this selection
    bt_sel = bt_sum[(bt_sum["country"] == country) & (bt_sum["indicator"] == indicator)].copy()

    if bt_sel.empty:
        st.warning("No backtesting metrics available for this country + indicator.")
        return

    bt_sel = bt_sel.sort_values("horizon")
    st.subheader("Horizon-wise Backtesting Metrics")

    st.dataframe(
        bt_sel[["horizon", "mae", "rmse", "mape_pct"]]
        .rename(columns={
            "horizon": "Horizon (years ahead)",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape_pct": "MAPE (%)"
        })
        .reset_index(drop=True)
    )

    # Decay curve: MAPE vs horizon
    st.subheader("Forecast Error Decay Curve (MAPE vs Horizon)")

    decay_chart = (
        alt.Chart(bt_sel)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:Q", title="Forecast horizon (years ahead)"),
            y=alt.Y("mape_pct:Q", title="MAPE (%)"),
            tooltip=["horizon", "mape_pct"]
        )
    )
    st.altair_chart(decay_chart, width='stretch')

    # Historical + forecast + uncertainty bands
    st.subheader("Forecast with Uncertainty Band")

    # historical
    if indicator not in hist.columns:
        st.error(f"Indicator {indicator} not found in historical data.")
        return

    hist_sel = hist[hist["country"] == country][["year", indicator]].dropna().copy()
    if hist_sel.empty:
        st.warning("No historical data for this selection.")
        return

    hist_sel = hist_sel.sort_values("year")
    hist_sel = hist_sel.rename(columns={indicator: "value"})
    hist_sel["type"] = "Historical"

    last_hist_year = hist_sel["year"].max()

    # forecast rows
    fc_sel = fc[(fc["country"] == country) & (fc["indicator"] == indicator)].copy()
    if fc_sel.empty:
        st.warning("No forecast rows for this selection.")
        return

    fc_sel = fc_sel.sort_values("year")
    fc_sel = fc_sel.rename(columns={"forecast_mean": "value"})
    fc_sel["type"] = "Forecast"

    # compute uncertainty band from backtesting
    fc_band = compute_uncertainty_band(fc_sel, bt_sel, last_hist_year)

    # For plotting: possibly rescale GDP to trillions for readability
    y_title = indicator
    scale_factor = 1.0

    if indicator == "gdp_current_usd":
        scale_factor = 1_000_000_000_000.0
        hist_sel["value"] = hist_sel["value"] / scale_factor
        fc_band["value"] = fc_band["value"] / scale_factor
        fc_band["lower"] = fc_band["lower"] / scale_factor
        fc_band["upper"] = fc_band["upper"] / scale_factor
        y_title = "GDP (Trillion USD)"

    combined = pd.concat(
        [
            hist_sel[["year", "value", "type"]],
            fc_band[["year", "value", "type"]],
        ],
        ignore_index=True,
    )

    # Band chart for forecast part
    band_chart = (
        alt.Chart(fc_band)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("lower:Q", title=y_title),
            y2="upper:Q",
            tooltip=["year", "value", "mape_pct"]
        )
    )

    # Line chart for historical + forecast
    line_chart = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("type:N", title="Series"),
            tooltip=["year", "type", "value"]
        )
    )

    st.altair_chart((band_chart + line_chart).interactive(), width='stretch')

    st.markdown(
        """
**Interpretation**:
- The solid line shows historical data and the model’s central forecast.
- The shaded region around the forecast is an *uncertainty band* derived from backtesting:
  for each forecast year, the average MAPE at that horizon (1Y, 2Y, 3Y, …) is applied as ±% around the point forecast.
- As the forecast horizon increases, the band usually widens, reflecting growing uncertainty.
"""
    )

    # Optional: show raw forecast values + band
    with st.expander("Show forecast values with uncertainty band"):
        show_cols = ["year", "value", "horizon", "mape_pct", "lower", "upper"]
        st.dataframe(fc_band[show_cols].rename(columns={"value": "forecast"}))


if __name__ == "__main__":
    main()
