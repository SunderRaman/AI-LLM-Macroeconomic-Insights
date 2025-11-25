# forecast_module.py
"""
Forecasting module (Option-C):
- Produces forecasts for multiple macro indicators for each country using Prophet.
- Applies log-transform for GDP-related indicators to stabilize forecasts (gdp_current_usd, gdp_per_capita).
- Leaves other indicators untransformed (inflation, unemployment, exports%, imports%, gdp_growth_pct, population).
- Outputs a single CSV: data/processed/forecasts/forecasts_all.csv with columns:
    country, indicator, year, forecast_mean

Usage:
    python src/forecasting/forecast_module.py

Notes:
- Requires prophet package to be installed (prophet / cmdstanpy configured).
- Reads historical data from data/processed/wb_macro_clean.csv
- Forecast horizon: by default 10 years beyond the latest historical year. The code
  produces forecasts for years [max_hist_year .. max_hist_year + horizon].

This script is written to be robust: it falls back to simple extension when data is
insufficient for modeling.
"""

import os
import warnings
from math import isnan
import numpy as np
import pandas as pd

# Prophet import
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
except Exception as e:
    raise ImportError("Prophet not found. Please install prophet (and cmdstanpy). Error: {}".format(e))

# Paths
WB_CLEAN = os.path.join("data", "processed", "wb_macro_clean.csv")
OUT_DIR = os.path.join("data", "processed", "forecasts")
OUT_PATH = os.path.join(OUT_DIR, "forecasts_all.csv")

# Config
FORECAST_HORIZON = 10  # years beyond the latest historical year (keeps inclusive of last year)
MIN_POINTS_TO_MODEL = 4  # minimum historical points to fit Prophet model
EPS = 1e-6  # small value for log-transform

# Indicators that should be log-transformed
LOG_INDICATORS = {"gdp_current_usd", "gdp_per_capita"}

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


def _prepare_prophet_df(series_df, value_col, date_year_col='year'):
    """Convert a simple year-indexed dataframe to Prophet's dataframe format (ds, y).
    series_df: DataFrame with columns [year, value_col]
    Returns DataFrame with ds (datetime) and y
    """
    df = series_df[[date_year_col, value_col]].dropna().copy()
    if df.empty:
        return df
    # Create a Jan-01 date for each year (Prophet handles freq as annual)
    df = df.rename(columns={date_year_col: 'year', value_col: 'y'})
    df['ds'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-01-01')
    # reorder
    df = df[['ds', 'y', 'year']]
    return df


def fit_and_forecast_prophet(series_df, years_to_forecast, log_transform=False):
    """
    Fit Prophet to series_df (columns year, value) and return forecast array aligned to years_to_forecast.
    If log_transform=True, the function fits log(y) and returns exp(predicted).
    Falls back to simple extension if insufficient data.
    """
    # Prepare df
    dfp = _prepare_prophet_df(series_df, series_df.columns[1], date_year_col=series_df.columns[0])
    if dfp.empty:
        return np.array([np.nan] * len(years_to_forecast))

    # Ensure numeric
    dfp['y'] = pd.to_numeric(dfp['y'], errors='coerce')
    dfp = dfp.dropna()

    if len(dfp) < MIN_POINTS_TO_MODEL:
        # fallback: extend last value
        last_val = dfp['y'].iloc[-1]
        if log_transform:
            # if log_transform, exponentiate after storing log
            return np.exp(np.repeat(np.log(max(last_val, EPS)), len(years_to_forecast)))
        return np.repeat(last_val, len(years_to_forecast))

    # apply log transform if required
    if log_transform:
        dfp['y'] = np.log(np.maximum(dfp['y'], EPS))

    # fit prophet
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    # allow changepoints but keep defaults
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m.fit(dfp[['ds', 'y']])
        except Exception as e:
            # fallback to extension if fit fails
            last_val = dfp['y'].iloc[-1]
            if log_transform:
                return np.exp(np.repeat(last_val, len(years_to_forecast)))
            return np.repeat(last_val, len(years_to_forecast))

    # create future dataframe for the requested years
    future_ds = pd.to_datetime([f"{int(y)}-01-01" for y in years_to_forecast])
    future = pd.DataFrame({'ds': future_ds})
    # predict
    fc = m.predict(future)
    yhat = fc['yhat'].values
    if log_transform:
        yhat = np.exp(yhat)
    return yhat


def generate_forecasts(wb_path=WB_CLEAN, horizon=FORECAST_HORIZON):
    """
    Main function: reads wb_clean, generates forecasts for specified indicators and countries.
    Returns DataFrame with columns: country, indicator, year, forecast_mean
    """
    wb = pd.read_csv(wb_path)
    # required columns
    required = {'country', 'year'}
    if not required.issubset(set(wb.columns)):
        raise KeyError(f"Input WB file missing required columns: {required}")

    # Identify indicators available (exclude country/year columns)
    indicator_cols = [c for c in wb.columns if c not in ['country', 'country_code', 'year']]

    # We'll forecast each (country, indicator) series
    out_rows = []

    # Determine common year grid
    max_hist_year = int(wb['year'].max())
    forecast_years = np.arange(max_hist_year, max_hist_year + horizon + 1)  # inclusive of last hist year

    for country, grp in wb.groupby('country'):
        grp_sorted = grp.sort_values('year')
        for indicator in indicator_cols:
            series = grp_sorted[['year', indicator]].dropna()
            # if the series is empty, skip
            if series.empty:
                continue

            # choose log_transform for GDP-related
            log_tf = indicator in LOG_INDICATORS

            # prepare input to model: rename columns so fit function can detect year/val
            series_for_model = series.copy()
            series_for_model.columns = ['year', 'value']

            # fit & forecast
            preds = fit_and_forecast_prophet(series_for_model, forecast_years, log_transform=log_tf)

            # assemble rows
            for y, val in zip(forecast_years, preds):
                out_rows.append({'country': country, 'indicator': indicator, 'year': int(y), 'forecast_mean': float(val) if not isnan(val) else np.nan})

    out_df = pd.DataFrame(out_rows)
    # sort
    out_df = out_df.sort_values(['country', 'indicator', 'year']).reset_index(drop=True)

    # Save to CSV
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved forecasts to: {OUT_PATH}")
    return out_df


if __name__ == '__main__':
    print("Generating forecasts (this may take a few minutes)...")
    df_out = generate_forecasts()
    print(df_out.head())
    print("Done.")
