"""
scenario_engine.py  (Option-1: NX% trend-based trade-aware scenarios)

Place at: src/forecasting/scenario_engine.py

Notes:
 - Expects baseline forecasts at:
       data/processed/forecasts/forecasts_all.csv
 - Expects cleaned WB data at:
       data/processed/wb_macro_clean.csv
   (if not present, the file /mnt/data/wb_macro_clean.csv is read if available)
 - Produces scenario CSVs in:
       data/processed/forecasts/scenarios/
 - GDP in saved CSVs is converted to Trillions (and header updated)
 - Trade sensitivity alpha (ALPHA_NX) set to 0.20 (changeable)
"""

import os
import numpy as np
import pandas as pd

# Config / paths
FORECASTS_PATH = os.path.join("data", "processed", "forecasts", "forecasts_all.csv")
WB_CLEAN_PRIMARY = os.path.join("data", "processed", "wb_macro_clean.csv")
WB_CLEAN_ALT = "/mnt/data/wb_macro_clean.csv"   # fallback if primary not present
SCENARIO_SAVE_DIR = os.path.join("data", "processed", "forecasts", "scenarios")

# Trade sensitivity
ALPHA_NX = 0.20

# Forecast NX trend params
TREND_YEARS = 10  # use last 10 years to estimate linear trend

# Safety caps
MAX_GDP_SHIFT_PCT = 0.05  # max ±5% of GDP per year from trade adjustment


# -------------------------
# Utilities - loaders
# -------------------------
def _load_baseline_forecasts():
    if not os.path.exists(FORECASTS_PATH):
        raise FileNotFoundError(f"Baseline forecasts not found: {FORECASTS_PATH}")
    df = pd.read_csv(FORECASTS_PATH)
    df['year'] = df['year'].astype(int)
    df['forecast_mean'] = pd.to_numeric(df['forecast_mean'], errors='coerce')
    return df


def _load_wb_clean():
    # prefer primary path, fallback to uploaded path
    path = WB_CLEAN_PRIMARY if os.path.exists(WB_CLEAN_PRIMARY) else WB_CLEAN_ALT
    if not os.path.exists(path):
        raise FileNotFoundError(f"WB cleaned CSV not found at {WB_CLEAN_PRIMARY} or {WB_CLEAN_ALT}")
    wb = pd.read_csv(path)
    # Expect cleaned column names: gdp_current_usd, exports_pct_gdp, imports_pct_gdp
    required = {'country', 'year', 'gdp_current_usd', 'exports_pct_gdp', 'imports_pct_gdp'}
    if not required.issubset(set(wb.columns)):
        raise KeyError(f"WB cleaned CSV missing required cols. Found: {wb.columns.tolist()}")
    # Keep only these
    wb = wb[['country', 'year', 'gdp_current_usd', 'exports_pct_gdp', 'imports_pct_gdp']].copy()
    # Ensure numeric types
    wb['year'] = wb['year'].astype(int)
    wb['gdp_current_usd'] = pd.to_numeric(wb['gdp_current_usd'], errors='coerce')
    wb['exports_pct_gdp'] = pd.to_numeric(wb['exports_pct_gdp'], errors='coerce')
    wb['imports_pct_gdp'] = pd.to_numeric(wb['imports_pct_gdp'], errors='coerce')
    # compute NX% (exports - imports)
    wb['nx_pct'] = wb['exports_pct_gdp'] - wb['imports_pct_gdp']
    return wb


# -------------------------
# NX% trend forecast helper
# -------------------------
def _forecast_nx_pct_for_years(nx_hist_df, target_years):
    """
    nx_hist_df: DataFrame with columns ['year','nx_pct'] for a country (historical)
    target_years: np.array/list of forecast years to produce nx_pct for
    Returns: numpy array of nx_pct for target_years (aligned order)
    """
    # If insufficient historical data, fallback to last known value
    nx_hist = nx_hist_df.dropna(subset=['nx_pct'])
    if nx_hist.empty:
        # no data -> zeros
        return np.zeros(len(target_years))
    if len(nx_hist) < 2:
        # not enough points for trend -> flat extension
        last = nx_hist['nx_pct'].iloc[-1]
        return np.repeat(last, len(target_years))

    # use last TREND_YEARS points for linear fit
    hist_tail = nx_hist.tail(TREND_YEARS)
    x = hist_tail['year'].values
    y = hist_tail['nx_pct'].values
    # robust linear fit (polyfit)
    m, b = np.polyfit(x, y, 1)
    trend_vals = m * target_years + b
    return trend_vals


# -------------------------
# Core scenario builders
# -------------------------
def build_baseline():
    base = _load_baseline_forecasts()
    out = base.rename(columns={'forecast_mean': 'scenario_forecast'})[['country', 'indicator', 'year', 'scenario_forecast']]
    return out


def build_trade_aware_nxpct():
    """
    Build scenario using NX% trend forecasting.
    Approach:
      - Forecast nx_pct for each country using linear trend on last TREND_YEARS
      - Baseline nx_usd = baseline_gdp * nx_pct_baseline / 100
      - Scenario nx_pct = nx_pct_baseline + delta_pct (policy bump; here +0.5 percentage point)
      - Compute nx_usd_scenario = baseline_gdp * nx_pct_scenario / 100
      - delta_usd = nx_usd_scenario - nx_usd_baseline
      - GDP_adj = baseline_gdp + ALPHA_NX * delta_usd
      - Cap ALPHA adjustment to ±MAX_GDP_SHIFT_PCT * baseline_gdp
    """
    baseline = _load_baseline_forecasts()
    wb = _load_wb_clean()

    out_list = []

    for (country, indicator), group in baseline.groupby(['country', 'indicator']):
        g = group.sort_values('year').reset_index(drop=True)

        if indicator != "gdp_current_usd":
            # keep other indicators unchanged
            out_list.append(g.rename(columns={'forecast_mean': 'scenario_forecast'})[['country','indicator','year','scenario_forecast']])
            continue

        years = g['year'].values
        # get historical nx_pct for this country
        nx_hist = wb[wb['country'] == country][['year','nx_pct']].sort_values('year')
        # baseline forecast of nx_pct (trend)
        nx_pct_baseline = _forecast_nx_pct_for_years(nx_hist, years)
        # scenario: small policy improvement -> add +0.5 percentage points to nx_pct over the horizon
        nx_pct_scenario = nx_pct_baseline + 0.5  # +0.5 pct point

        # baseline GDP (USD) from forecasts
        gdp_baseline = g['forecast_mean'].values  # array

        # compute NX in USD for baseline and scenario
        nx_usd_baseline = gdp_baseline * (nx_pct_baseline / 100.0)
        nx_usd_scenario = gdp_baseline * (nx_pct_scenario / 100.0)

        # delta (USD) and ALPHA adjustment
        delta_usd = nx_usd_scenario - nx_usd_baseline
        delta_adj = ALPHA_NX * delta_usd

        # cap adjustments to ±MAX_GDP_SHIFT_PCT * baseline_gdp
        max_shift = MAX_GDP_SHIFT_PCT * gdp_baseline
        delta_capped = np.clip(delta_adj, -max_shift, max_shift)

        # final adjusted GDP (scenario)
        scenario_vals = gdp_baseline + delta_capped

        g_out = g[['country','indicator','year']].copy()
        g_out['scenario_forecast'] = scenario_vals
        out_list.append(g_out)

    return pd.concat(out_list, ignore_index=True)


def build_recession_scenario():
    """
    Recession scenario:
     - nx_pct trend baseline, but apply a temporary negative shock:
       reduce nx_pct by 1.0 percentage point for first 2 forecast years (then revert to baseline)
    """
    baseline = _load_baseline_forecasts()
    wb = _load_wb_clean()

    out_list = []

    for (country, indicator), group in baseline.groupby(['country', 'indicator']):
        g = group.sort_values('year').reset_index(drop=True)

        if indicator != "gdp_current_usd":
            out_list.append(g.rename(columns={'forecast_mean':'scenario_forecast'})[['country','indicator','year','scenario_forecast']])
            continue

        years = g['year'].values
        nx_hist = wb[wb['country']==country][['year','nx_pct']].sort_values('year')
        nx_pct_baseline = _forecast_nx_pct_for_years(nx_hist, years)

        # create scenario nx_pct with a -1.0 percentage point shock for first 2 years
        nx_pct_scenario = nx_pct_baseline.copy()
        if len(nx_pct_scenario) >= 1:
            nx_pct_scenario[0] = nx_pct_scenario[0] - 1.0
        if len(nx_pct_scenario) >= 2:
            nx_pct_scenario[1] = nx_pct_scenario[1] - 0.5  # partial recovery year

        gdp_baseline = g['forecast_mean'].values
        nx_usd_baseline = gdp_baseline * (nx_pct_baseline / 100.0)
        nx_usd_scenario = gdp_baseline * (nx_pct_scenario / 100.0)

        delta_usd = nx_usd_scenario - nx_usd_baseline
        delta_adj = ALPHA_NX * delta_usd
        max_shift = MAX_GDP_SHIFT_PCT * gdp_baseline
        delta_capped = np.clip(delta_adj, -max_shift, max_shift)
        scenario_vals = gdp_baseline + delta_capped

        g_out = g[['country','indicator','year']].copy()
        g_out['scenario_forecast'] = scenario_vals
        out_list.append(g_out)

    return pd.concat(out_list, ignore_index=True)


# -------------------------
# Master preset generator
# -------------------------
def generate_all_presets():
    presets = {
        "Baseline": build_baseline(),
        "Trade-Aware (α=0.20)": build_trade_aware_nxpct(),
        "Recession (-1.0 pct shock)": build_recession_scenario()
    }
    return presets


# -------------------------
# CSV save utilities (GDP -> Trillions, rename header)
# -------------------------
def save_scenarios_to_csv(scenarios_dict, directory=SCENARIO_SAVE_DIR, save_combined=True):
    """
    scenarios_dict: dict of {scenario_name: dataframe}
    directory: output dir
    If save_combined=True, writes a combined CSV (all_scenarios_combined.csv)
    GDP series are converted to trillions and the column renamed:
        "scenario_forecast (in $Trillion)"
    Non-GDP indicators keep column name "scenario_forecast"
    """
    os.makedirs(directory, exist_ok=True)
    combined_list = []

    for name, df in scenarios_dict.items():
        df_out = df.copy()

        # If GDP present in this df, convert to trillions and rename column
        if "gdp_current_usd" in df_out['indicator'].unique():
            mask = df_out['indicator'] == 'gdp_current_usd'
            df_out.loc[mask, 'scenario_forecast'] = df_out.loc[mask, 'scenario_forecast'] / 1_000_000_000_000.0
            # df_out = df_out.rename(columns={'scenario_forecast': 'scenario_forecast (in $Trillion)'})

        # safe file name
        fname = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        fpath = os.path.join(directory, fname)
        df_out.to_csv(fpath, index=False)
        combined_list.append(df_out.assign(scenario=name))
        print(f"Saved: {fpath}")

    if save_combined:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_path = os.path.join(directory, "all_scenarios_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined scenarios CSV: {combined_path}")

    return directory


# -------------------------
# Quick CLI test helper
# -------------------------
if __name__ == "__main__":
    print("Generating presets (this may take a few seconds)...")
    presets = generate_all_presets()
    save_dir = save_scenarios_to_csv(presets)
    print("Done. Files written to:", save_dir)
    # df = pd.read_csv("data/processed/forecasts/forecasts_all.csv")
    # g = df[(df.country=="Germany") & (df.indicator=="gdp_current_usd")]
    # print(g)
