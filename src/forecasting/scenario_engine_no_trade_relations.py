"""
scenario_engine.py

Simple scenario engine that:
- Loads baseline forecasts (forecasts_all.csv)
- Produces variants by applying:
  * growth boosts (additive % points to annual growth)
  * multiplicative scaling (geometric) to hit a target year value (e.g., India $5T)
  * inflation shocks (applied to inflation indicator)
  * currency shocks (modeled as multiplicative reduction in USD-GDP)
- Returns scenario DataFrames (country, indicator, year, forecast)
"""

import os
import pandas as pd
import numpy as np

FORECASTS_PATH = os.path.join("data", "processed", "forecasts", "forecasts_all.csv")

def load_baseline():
    """Load baseline forecasts (as created by forecasting.py)."""
    df = pd.read_csv(FORECASTS_PATH)
    # ensure types
    df['year'] = df['year'].astype(int)
    df['forecast_mean'] = pd.to_numeric(df['forecast_mean'], errors='coerce')
    return df

# -------------------------
# Utility helpers
# -------------------------
def apply_growth_boost(df_country_indicator, annual_boost_pct):
    """
    Apply a constant additional nominal growth rate (annual_boost_pct as decimal, e.g., 0.02 = +2%)
    This treats df_country_indicator as baseline levels (not growth rates).
    It multiplies each future year by (1 + annual_boost_pct)^(year_offset).
    """
    df = df_country_indicator.copy().sort_values('year').reset_index(drop=True)
    years = df['year'].values
    start_year = years.min()
    # compute offsets from start (0 for first forecast year)
    offsets = years - start_year
    df['scenario_forecast'] = df['forecast_mean'] * ((1.0 + annual_boost_pct) ** offsets)
    return df[['country','indicator','year','scenario_forecast']]

def scale_to_target(df_country_indicator, target_year, target_value):
    """
    Scale the forecast series geometrically so that the value for target_year equals target_value.
    We compute a multiplicative factor per year such that shape is preserved but level shifts.
    This is useful for 'India $5T by 2029' adjustments.
    """
    df = df_country_indicator.copy().sort_values('year').reset_index(drop=True)
    if target_year not in df['year'].values:
        raise ValueError("target_year not in baseline forecast years")
    baseline_target = df.loc[df['year'] == target_year, 'forecast_mean'].values[0]
    if baseline_target <= 0 or np.isnan(baseline_target):
        raise ValueError("baseline target value not valid")
    # overall multiplier needed at target year
    multiplier_needed = float(target_value) / float(baseline_target)
    # apply geometric scaling across years by raising multiplier_needed to (offset/offset_target)
    years = df['year'].values
    offset_target = target_year - years.min()
    if offset_target <= 0:
        # if target is at start, apply uniform scale
        df['scenario_forecast'] = df['forecast_mean'] * multiplier_needed
        return df[['country','indicator','year','scenario_forecast']]

    # compute per-year multipliers that compound to multiplier_needed at target
    # per_year_multiplier = multiplier_needed ** (1/offset_target)
    per_year_multiplier = multiplier_needed ** (1.0 / offset_target)
    offsets = years - years.min()
    df['scenario_forecast'] = df['forecast_mean'] * (per_year_multiplier ** offsets)
    return df[['country','indicator','year','scenario_forecast']]

def apply_currency_shock(df_country_indicator, annual_fx_depreciation_pct):
    """
    Model an FX shock as a multiplicative reduction in USD-denominated GDP.
    annual_fx_depreciation_pct is decimal e.g., 0.03 means 3% weaker local currency each year -> USD value lower.
    We multiply baseline by (1 - annual_fx_depreciation_pct) ** offsets.
    """
    df = df_country_indicator.copy().sort_values('year').reset_index(drop=True)
    years = df['year'].values
    offsets = years - years.min()
    df['scenario_forecast'] = df['forecast_mean'] * ((1.0 - annual_fx_depreciation_pct) ** offsets)
    return df[['country','indicator','year','scenario_forecast']]

# -------------------------
# Scenario builders (presets)
# -------------------------
def build_baseline():
    """Return baseline forecast df (renamed scenario_forecast = baseline)"""
    base = load_baseline()
    out = base.rename(columns={'forecast_mean':'scenario_forecast'}).copy()
    return out[['country','indicator','year','scenario_forecast']]

def build_optimistic_option_c(target_country="India", target_indicator="gdp_current_usd",
                             target_year=2029, target_value=None,
                             growth_boost_pct=0.02, fx_improvement_pct=0.01):
    """
    Option C: balanced (small extra nominal growth + small FX improvement) +
    optional direct scaling to hit a target_value for the target_country/indicator.
    If target_value is provided (e.g., 5e12 for $5T), we scale to hit it.
    Otherwise, we apply growth_boost_pct and reduce currency depreciation by applying FX improvement.
    """
    base = load_baseline()
    out_list = []

    # iterate countries/indicators
    for (country, indicator), group in base.groupby(['country','indicator']):
        g = group.sort_values('year').reset_index(drop=True)
        if country == target_country and indicator == target_indicator and target_value is not None:
            # scale geometry to hit target
            try:
                sc = scale_to_target(g, target_year, target_value)
                out_list.append(sc)
                continue
            except Exception as e:
                # fallback to growth boost if scale fails
                pass

        # apply small growth boost to GDP levels only (not percentage indicators)
        if indicator == "gdp_current_usd":
            sc = apply_growth_boost(g, annual_boost_pct=growth_boost_pct)
            # apply small FX improvement by reducing an effective depreciation
            # Implementation: treat fx_improvement_pct as negative depreciation: multiply by (1 + fx_improvement_pct)^offset
            years = sc['year'].values
            offsets = years - years.min()
            sc['scenario_forecast'] = sc['scenario_forecast'] * ((1.0 + fx_improvement_pct) ** offsets)
            out_list.append(sc)
        elif indicator == "inflation_cpi_pct":
            # a modest policy improvement -> slightly lower inflation relative to baseline
            # subtract a small absolute percentage (not below zero)
            sc = g.copy()
            sc['scenario_forecast'] = (sc['forecast_mean'] - (growth_boost_pct * 100 * 0.2)).clip(lower=0)
            out_list.append(sc[['country','indicator','year','scenario_forecast']])
        else:
            # other indicators: keep baseline
            out_list.append(g.rename(columns={'forecast_mean':'scenario_forecast'})[['country','indicator','year','scenario_forecast']])

    return pd.concat(out_list, ignore_index=True)

def build_recession_scenario(global_shock_pct=-0.05, duration_years=2):
    """
    Apply a negative shock to GDP levels for `duration_years` starting at the first forecast year.
    global_shock_pct is negative (e.g., -0.05 => -5% instantaneous reduction applied in year 1, then partial recovery).
    We'll apply: year0 * (1 + shock), year1 * (1 + shock/2), etc.
    """
    base = load_baseline()
    out_list = []
    for (country, indicator), group in base.groupby(['country','indicator']):
        g = group.sort_values('year').reset_index(drop=True)
        if indicator == "gdp_current_usd":
            years = g['year'].values
            offsets = years - years.min()
            scen = g.copy()
            scen_vals = []
            for off, val in zip(offsets, scen['forecast_mean'].values):
                if off == 0:
                    scen_vals.append(val * (1.0 + global_shock_pct))
                elif 0 < off < duration_years:
                    scen_vals.append(val * (1.0 + global_shock_pct * 0.5))
                else:
                    scen_vals.append(val)  # revert to baseline after shock period
            scen['scenario_forecast'] = scen_vals
            out_list.append(scen[['country','indicator','year','scenario_forecast']])
        else:
            out_list.append(g.rename(columns={'forecast_mean':'scenario_forecast'})[['country','indicator','year','scenario_forecast']])
    return pd.concat(out_list, ignore_index=True)

def build_currency_depreciation_scenario(annual_depreciation_pct=0.03):
    base = load_baseline()
    out_list = []
    for (country, indicator), group in base.groupby(['country','indicator']):
        g = group.sort_values('year').reset_index(drop=True)
        if indicator == "gdp_current_usd":
            sc = apply_currency_shock(g, annual_depreciation_pct)
            out_list.append(sc)
        else:
            out_list.append(g.rename(columns={'forecast_mean':'scenario_forecast'})[['country','indicator','year','scenario_forecast']])
    return pd.concat(out_list, ignore_index=True)

# -------------------------
# Convenience function to gather scenarios
# -------------------------
def generate_all_presets(india_target_5t=False):
    """
    Generate a dict of scenario_name -> dataframe
    If india_target_5t=True, the optimistic option will attempt to force India to $5T by 2029.
    """
    baseline = build_baseline()
    presets = {"Baseline": baseline}

    if india_target_5t:
        optimistic = build_optimistic_option_c(target_country="India",
                                               target_indicator="gdp_current_usd",
                                               target_year=2029,
                                               target_value=5e12,
                                               growth_boost_pct=0.02, fx_improvement_pct=0.01)
    else:
        optimistic = build_optimistic_option_c(growth_boost_pct=0.02, fx_improvement_pct=0.01)

    presets["Optimistic (Option C)"] = optimistic
    presets["Recession (-5% shock)"] = build_recession_scenario(global_shock_pct=-0.05, duration_years=2)
    presets["Currency Depreciation (3% pa)"] = build_currency_depreciation_scenario(0.03)

    return presets

# Example usage:
# df_base = load_baseline()
# presets = generate_all_presets(india_target_5t=False)
if __name__ == "__main__":
    # Example: generate and print summary of scenarios
    scenarios = generate_all_presets(india_target_5t=True)
    save_dir = "data/processed/forecasts/scenarios"
    os.makedirs(save_dir, exist_ok=True)

    for name, df in scenarios.items():
        print(f"Scenario: {name}")
        fname = name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        fpath = os.path.join(save_dir, f"{fname}.csv")

        df.to_csv(fpath, index=False)
        print(f"Saved: {fpath}")