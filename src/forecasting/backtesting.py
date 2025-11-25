"""
backtesting.py

Module-D: Multi-horizon Backtesting & Diagnostics for macro forecasts.

- Rolling-origin (walk-forward) backtesting for multiple horizons (e.g. 1Y, 2Y, 3Y ahead)
- Uses Prophet with log-transform for GDP-like indicators (gdp_current_usd, gdp_per_capita)
- Computes MAE, RMSE, MAPE per (country, indicator, horizon, split)
- Saves:
    - backtest_metrics_multi.csv        (per split, per horizon)
    - backtest_metrics_by_horizon.csv   (mean per country/indicator/horizon)
    - backtest_details_multi.csv        (per year, per horizon)
- Plots:
    - bt_last_split_<country>_<indicator>.png (actual vs forecast for last split, horizon max)
    - bt_decay_<country>_<indicator>.png      (MAPE vs horizon curve)

Place at: src/forecasting/backtesting.py
"""

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prophet
try:
    from prophet import Prophet
except Exception:
    raise ImportError("Prophet required for backtesting. Please install prophet & cmdstanpy.")

# Paths
WB_CLEAN = os.path.join("data", "processed", "wb_macro_clean.csv")
OUT_DIR = os.path.join("data", "processed", "backtesting")
PLOTS_DIR = os.path.join("outputs", "plots", "backtesting")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Config
LOG_INDICATORS = {"gdp_current_usd", "gdp_per_capita"}
MIN_POINTS_TO_MODEL = 6      # minimum years of history to even try
MAX_ROLLING_SPLITS   = 4     # number of rolling-origin splits
HORIZONS_DEFAULT     = [1,2,3]   # multi-horizon backtest (1Y, 2Y, 3Y)


# ---- Metrics ----
def mae(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not mask.any():
        return np.nan
    return np.mean(np.abs(a[mask] - b[mask]))

def rmse(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not mask.any():
        return np.nan
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

def mape(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b) & (np.abs(b) > 1e-9)
    if not mask.any():
        return np.nan
    return np.mean(np.abs((a[mask] - b[mask]) / b[mask])) * 100.0


# ---- Helper: prepare Prophet input ----
def _to_prophet_df(years, values):
    df = pd.DataFrame({'year': years, 'y': values})
    df = df.dropna()
    if df.empty:
        return df
    df['ds'] = pd.to_datetime(df['year'].astype(int).astype(str) + "-01-01")
    return df[['ds','y','year']]


def fit_prophet_and_forecast(years_train, vals_train, target_years, log_transform=False):
    """
    Train Prophet on (years_train, vals_train) and forecast for all target_years.
    Returns predictions aligned with target_years (numpy array).
    """
    dfp = _to_prophet_df(np.array(years_train, dtype=int),
                         np.array(vals_train, dtype=float))
    if dfp.shape[0] < 2:
        return np.array([np.nan]*len(target_years))

    if log_transform:
        dfp['y'] = np.log(np.maximum(dfp['y'], 1e-6))

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    try:
        m.fit(dfp[['ds','y']])
    except Exception:
        # fallback: flat extension
        last = dfp['y'].iloc[-1]
        if log_transform:
            return np.exp(np.repeat(last, len(target_years)))
        return np.repeat(last, len(target_years))

    future = pd.DataFrame({
        'ds': pd.to_datetime([f"{int(y)}-01-01" for y in target_years])
    })
    fc = m.predict(future)
    yhat = fc['yhat'].values
    if log_transform:
        yhat = np.exp(yhat)
    return yhat


# ---- Multi-horizon backtest for one (country, indicator) ----
def backtest_series_multi(country, indicator, hist_df,
                          horizons=HORIZONS_DEFAULT,
                          min_train=MIN_POINTS_TO_MODEL,
                          max_splits=MAX_ROLLING_SPLITS):
    """
    hist_df: DataFrame with columns ['year','value'], sorted by year.
    horizons: list of forecast horizons, e.g. [1,2,3]
    Returns:
      metrics_df: rows per (split, horizon)
      details_df: rows per (split, horizon, year)
    """
    hist_df = hist_df.dropna().sort_values('year')
    years  = hist_df['year'].values
    values = hist_df['value'].values
    n = len(years)

    max_h = max(horizons)
    if n < min_train + max_h:
        return None, None

    # Determine rolling splits: choose up to max_splits training end indices
    max_end = n - max_h
    possible_ends = np.arange(min_train, max_end + 1)
    if len(possible_ends) <= max_splits:
        ends = possible_ends
    else:
        idx = np.linspace(0, len(possible_ends)-1, max_splits, dtype=int)
        ends = possible_ends[idx]

    all_rows = []
    all_details = []

    log_tf = indicator in LOG_INDICATORS

    for end in ends:
        train_years = years[:end]
        train_vals  = values[:end]
        # forecast for all years up to max_h ahead
        target_years_full = years[end:end+max_h]   # max_h future points
        preds_full = fit_prophet_and_forecast(train_years, train_vals, target_years_full, log_transform=log_tf)

        # For each horizon h, evaluate first h steps of preds_full vs actuals
        for h in horizons:
            if end + h > n:
                continue  # not enough data to evaluate this horizon at this split

            test_years  = years[end:end+h]
            test_actual = values[end:end+h]
            test_pred   = preds_full[:h]

            m_mae  = mae(test_pred, test_actual)
            m_rmse = rmse(test_pred, test_actual)
            m_mape = mape(test_pred, test_actual)

            all_rows.append({
                'country': country,
                'indicator': indicator,
                'horizon': h,
                'train_end_year': int(train_years[-1]),
                'test_years': f"{test_years[0]}-{test_years[-1]}",
                'mae': float(m_mae),
                'rmse': float(m_rmse),
                'mape_pct': float(m_mape),
                'n_train': len(train_years),
                'n_test': len(test_years)
            })

            for y, act, pred in zip(test_years, test_actual, test_pred):
                all_details.append({
                    'country': country,
                    'indicator': indicator,
                    'horizon': h,
                    'train_end_year': int(train_years[-1]),
                    'year': int(y),
                    'actual': float(act),
                    'forecast': float(pred),
                    'error': float(pred - act)
                })

    if not all_rows:
        return None, None

    metrics_df = pd.DataFrame(all_rows)
    details_df = pd.DataFrame(all_details)
    return metrics_df, details_df


# ---- Top-level runner ----
def run_backtesting_multi(wb_path=WB_CLEAN,
                          out_dir=OUT_DIR,
                          plots_dir=PLOTS_DIR,
                          horizons=HORIZONS_DEFAULT):
    print("Running multi-horizon backtesting...")

    wb = pd.read_csv(wb_path)
    wb['year'] = wb['year'].astype(int)

    indicator_cols = [c for c in wb.columns if c not in ['country','country_code','year']]

    all_metrics = []
    all_details = []

    for country, grp in wb.groupby('country'):
        print(f"Backtesting {country} ...")
        grp_sorted = grp.sort_values('year')
        for ind in indicator_cols:
            series = grp_sorted[['year', ind]].dropna().rename(columns={ind: 'value'})
            if series.empty or len(series) < MIN_POINTS_TO_MODEL + max(horizons):
                continue

            metrics_df, details_df = backtest_series_multi(
                country, ind, series,
                horizons=horizons,
                min_train=MIN_POINTS_TO_MODEL,
                max_splits=MAX_ROLLING_SPLITS
            )
            if metrics_df is None:
                continue

            all_metrics.append(metrics_df)
            all_details.append(details_df)

            # Small plot: last split, max horizon actual vs forecast
            try:
                last_train_end = metrics_df['train_end_year'].max()
                h_max = max(horizons)
                subset = details_df[(details_df['train_end_year']==last_train_end) &
                                    (details_df['horizon']==h_max)]
                hist_years = series['year'].values
                hist_vals  = series['value'].values
                fig, ax = plt.subplots(figsize=(7,3.5))
                ax.plot(hist_years, hist_vals, label='Historical', linewidth=1)
                ax.scatter(subset['year'], subset['actual'], label='Actual (test)', color='black')
                ax.scatter(subset['year'], subset['forecast'], label=f'Forecast (h={h_max})', color='red', marker='x')
                ax.set_title(f"{country} — {ind} (last split, horizon {h_max})")
                ax.set_xlabel("Year")
                ax.set_ylabel(ind)
                ax.legend()
                fname = f"bt_last_split_{country.replace(' ','_')}_{ind}.png"
                fpath = os.path.join(plots_dir, fname)
                fig.savefig(fpath, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception:
                pass

    # Combine metrics & details
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_path = os.path.join(out_dir, "backtest_metrics_multi.csv")
        metrics_all.to_csv(metrics_path, index=False)
        print("Saved multi-horizon metrics:", metrics_path)
    else:
        metrics_all = None

    if all_details:
        details_all = pd.concat(all_details, ignore_index=True)
        details_path = os.path.join(out_dir, "backtest_details_multi.csv")
        details_all.to_csv(details_path, index=False)
        print("Saved multi-horizon detailed errors:", details_path)
    else:
        details_all = None

    # Aggregate by horizon: mean metrics per (country, indicator, horizon)
    if metrics_all is not None:
        agg = (metrics_all
               .groupby(['country','indicator','horizon'], as_index=False)
               [['mae','rmse','mape_pct']]
               .mean())
        agg_path = os.path.join(out_dir, "backtest_metrics_by_horizon.csv")
        agg.to_csv(agg_path, index=False)
        print("Saved metrics aggregated by horizon:", agg_path)

        # Decay curves: MAPE vs horizon for each (country, indicator)
        for (country, ind), g in agg.groupby(['country','indicator']):
            g_sorted = g.sort_values('horizon')
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(g_sorted['horizon'], g_sorted['mape_pct'],
                    marker='o', linestyle='-')
            ax.set_title(f"MAPE vs Horizon — {country}, {ind}")
            ax.set_xlabel("Forecast horizon (years)")
            ax.set_ylabel("MAPE (%)")
            ax.grid(True, alpha=0.3)
            fname = f"bt_decay_{country.replace(' ','_')}_{ind}.png"
            fpath = os.path.join(plots_dir, fname)
            fig.savefig(fpath, bbox_inches='tight', dpi=150)
            plt.close(fig)
    print("Multi-horizon backtesting finished.")


if __name__ == "__main__":
    # Default run: horizons 1,2,3
    run_backtesting_multi(horizons=[1,2,3])
