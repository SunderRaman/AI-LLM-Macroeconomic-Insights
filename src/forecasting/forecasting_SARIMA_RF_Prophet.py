# forecasting.py
"""
Module 4: Forecasting (GDP level and Inflation)
Place this file at: src/forecasting/forecasting.py

Outputs:
 - CSV forecasts: data/processed/forecasts/<country>_<indicator>_forecast.csv
 - PNG plots: outputs/plots/forecast_<country>_<indicator>.png
 - Combined CSV: data/processed/forecasts_all.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# statsmodels SARIMAX
import statsmodels.api as sm

# Optional libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Paths
PROC_PATH = "data/processed"
FORECAST_PATH = os.path.join(PROC_PATH, "forecasts")
PLOT_PATH = "outputs/plots"
os.makedirs(FORECAST_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# Configuration
START_YEAR = 2000
END_YEAR = 2024
FORECAST_START = END_YEAR + 1
FORECAST_HORIZON = 10   # default: forecast 10 years (2025-2034)
COUNTRIES = ["India", "United States", "China", "United Kingdom", "Germany", "Japan"]

# Indicators to forecast (World Bank cleaned names)
INDICATORS = {
    "gdp_current_usd": {"freq": "Y", "type": "level"},
    "inflation_cpi_pct": {"freq": "Y", "type": "rate"}
}

# -------------------------
# Utility functions
# -------------------------
def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_lag_features(series, lags=[1,2,3,4,5]):
    """
    series: pandas Series indexed by increasing time (year)
    returns DataFrame of lag features aligned with target
    """
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)
    df = df.dropna()
    return df

# -------------------------
# SARIMAX forecast function
# -------------------------
def forecast_sarimax(series, steps=10, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    series: pd.Series indexed by year (int) or datetime
    returns: forecast numpy array length=steps
    """
    try:
        # If index is year ints convert to datetime for SARIMAX
        if series.index.dtype == "int64" or series.index.dtype == "float64":
            idx = pd.to_datetime(series.index.astype(int).astype(str) + "-01-01")
            series = pd.Series(series.values, index=idx)

        model = sm.tsa.SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        fc = pred.predicted_mean
        # convert index back to integer years
        years = [int(d.year) for d in fc.index]
        return pd.Series(fc.values, index=years)
    except Exception as e:
        print("SARIMAX error:", e)
        return pd.Series([np.nan]*steps, index=range(FORECAST_START, FORECAST_START+steps))

# -------------------------
# RandomForest (lag-features) forecast
# -------------------------
def forecast_rf(series, steps=10, n_estimators=200):
    """
    Fit RandomForest on lag features and iteratively forecast steps ahead.
    series: pd.Series indexed by year ints
    """
    s = series.dropna()
    if s.shape[0] < 10:
        # not enough data
        return pd.Series([np.nan]*steps, index=range(FORECAST_START, FORECAST_START+steps))

    # create lag features
    lags = [1,2,3,4,5]
    df_lag = make_lag_features(s, lags=lags)

    X = df_lag.drop(columns=["y"]).values
    y = df_lag["y"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)

    # iterative forecasting
    last_vals = list(s.values[-max(lags):])  # most recent values
    preds = []
    for step in range(steps):
        # build feature vector
        feat = []
        for l in lags:
            if l <= len(last_vals):
                feat.append(last_vals[-l])
            else:
                feat.append(np.nan)
        feat = np.array(feat).reshape(1, -1)
        next_pred = model.predict(feat)[0]
        preds.append(next_pred)
        last_vals.append(next_pred)

    years = list(range(FORECAST_START, FORECAST_START+steps))
    return pd.Series(preds, index=years)

# -------------------------
# Prophet forecast wrapper (optional)
# -------------------------
def forecast_prophet(series, steps=10):
    if not PROPHET_AVAILABLE:
        return pd.Series([np.nan]*steps, index=range(FORECAST_START, FORECAST_START+steps))
    df = series.reset_index()
    df.columns = ["ds", "y"]
    # convert year ints to datetime
    if df["ds"].dtype != "datetime64[ns]":
        df["ds"] = pd.to_datetime(df["ds"].astype(str) + "-01-01")
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=steps, freq='Y')
    fc = m.predict(future)
    fc = fc.tail(steps)
    years = [int(d.year) for d in fc["ds"].dt.to_pydatetime()]
    return pd.Series(fc["yhat"].values, index=years)

# -------------------------
# Ensemble forecasting (average of available models)
# -------------------------
def ensemble_forecasts(series, steps=FORECAST_HORIZON):
    results = {}

    sar = forecast_sarimax(series, steps=steps)
    results["sarimax"] = sar

    rf = forecast_rf(series, steps=steps)
    results["rf"] = rf

    if PROPHET_AVAILABLE:
        prop = forecast_prophet(series, steps=steps)
        results["prophet"] = prop

    # Optional XGBoost (if installed) could be added similarly

    # Combine the series by aligning indexes and averaging (skip nans)
    df_fc = pd.DataFrame(results)
    avg = df_fc.mean(axis=1, skipna=True)
    return df_fc, avg

# -------------------------
# Plot & Save
# -------------------------
def plot_and_save(history_years, history_vals, forecast_series_dict, avg_series, country, indicator):
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(history_years, history_vals, label="Historical", color="black")
    # plot model forecasts
    for k, s in forecast_series_dict.items():
        ax.plot(s.index, s.values, linestyle='--', label=k)
    # ensemble avg
    ax.plot(avg_series.index, avg_series.values, linestyle='-', label='ensemble_avg', linewidth=2, color='red')

    ax.axvline(x=END_YEAR, color='gray', linestyle=':', label='Forecast start')
    ax.set_title(f"{country} — Forecast for {indicator}")
    ax.set_xlabel("Year")
    ax.set_ylabel(indicator)
    ax.legend(loc='best')
    plt.grid(True)
    fname = f"forecast_{country.replace(' ','_')}_{indicator}.png"
    path = os.path.join(PLOT_PATH, fname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved plot:", path)

def save_forecast_csv(country, indicator, avg_series, model_series_dict):
    df = pd.DataFrame({
        "year": avg_series.index
    })
    df["forecast_mean"] = avg_series.values
    # add model forecasts as columns
    for k, s in model_series_dict.items():
        df[k] = s.values
    outname = f"{country.replace(' ','_')}_{indicator}_forecast.csv"
    outpath = os.path.join(FORECAST_PATH, outname)
    df.to_csv(outpath, index=False)
    print("Saved forecast CSV:", outpath)
    return outpath

# -------------------------
# Main pipeline per country & indicator
# -------------------------
def run_forecasting():
    print("\n=== MODULE 4: Forecasting Started ===")

    wb = pd.read_csv(os.path.join(PROC_PATH, "wb_macro_clean.csv"))
    # Make sure columns exist
    wb.columns = [c.strip() for c in wb.columns]

    # Ensure numeric
    wb = ensure_numeric(wb, list(INDICATORS.keys()))

    all_forecasts = []

    for country in COUNTRIES:
        print(f"\n--- Forecasting for {country} ---")
        df_country = wb[wb["country"] == country].copy()
        if df_country.empty:
            print("No data for", country)
            continue

        for ind in INDICATORS.keys():
            if ind not in df_country.columns:
                print(f"Indicator {ind} not found in data; skipping.")
                continue

            # build series indexed by year (int)
            s = df_country.set_index("year")[ind].sort_index()
            s.index = s.index.astype(int)  # ensure proper index type
            
            # drop NA if any
            series = s.loc[START_YEAR:END_YEAR].dropna()
            series.index = pd.date_range(start=f"{START_YEAR}-01-01", periods=len(series), freq="YS")
            if series.shape[0] < 8:
                print(f"Not enough data points for {country} {ind} -> skipping")
                continue

            model_series_df, avg = ensemble_forecasts(series, steps=FORECAST_HORIZON)

            if ind == "gdp_current_usd":
            # 3-year rolling mean smoothing to eliminate noise/oscillation
                avg = avg.rolling(3, min_periods=1).mean()

            # Save results
            save_path = save_forecast_csv(country, ind, avg, model_series_df.to_dict(orient="series"))
            plot_and_save(series.index.values, series.values, model_series_df.to_dict(orient="series"), avg, country, ind)

            # collect for combined CSV
            temp = pd.DataFrame({
                "country": country,
                "indicator": ind,
                "year": avg.index,
                "forecast_mean": avg.values
            })
            all_forecasts.append(temp)

    # combine all
    if all_forecasts:
        all_df = pd.concat(all_forecasts, ignore_index=True)
        all_out = os.path.join(FORECAST_PATH, "forecasts_all.csv")
        all_df.to_csv(all_out, index=False)
        print("Saved combined forecasts:", all_out)

    print("\n=== MODULE 4: Forecasting Completed ===")

if __name__ == "__main__":
    run_forecasting()
