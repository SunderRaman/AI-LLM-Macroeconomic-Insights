import streamlit as st
import pandas as pd
import altair as alt
import sys, os

# Add project root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.forecasting.scenario_engine import (
    generate_all_presets,
    _load_baseline_forecasts
)

st.title("Scenario Forecasting")

# Load baseline + default scenarios
baseline = _load_baseline_forecasts()
presets = generate_all_presets(india_target_5t=False)
scenario_names = list(presets.keys())

# UI controls
col1, col2 = st.columns([3,1])

with col1:
    chosen = st.selectbox("Select scenario", scenario_names)

with col2:
    force_5t = st.checkbox("Force India $5T by 2029", value=False)

# If forcing India $5T, regenerate scenarios
if force_5t:
    presets = generate_all_presets(india_target_5t=True)
    scenario_names = list(presets.keys())
    chosen = "Optimistic (Option C)"  # auto-select optimistic

# Reload selected scenario
scen_df = presets[chosen]

# Country + indicator selection
countries = baseline['country'].unique().tolist()
indicators = baseline['indicator'].unique().tolist()

country = st.selectbox("Country", countries)
indicator = st.selectbox("Indicator", indicators)

# Filter scenario
sel = scen_df[
    (scen_df['country'] == country) &
    (scen_df['indicator'] == indicator)
].sort_values('year')

if sel.empty:
    st.warning("No data for this selection under chosen scenario")

else:
    # Load historical
    try:
        hist = pd.read_csv("data/processed/wb_macro_clean.csv")
        hist_sel = hist[hist['country'] == country][['year', indicator]].dropna()
        hist_sel = hist_sel.rename(columns={indicator: 'value'}).assign(type='Historical')
    except Exception:
        hist_sel = pd.DataFrame(columns=['year','value','type'])

    # Scenario forecast
    fc_sel = sel.rename(columns={'scenario_forecast': 'value'}).assign(type=chosen)

    # Combine for plotting
    combined = pd.concat([hist_sel[['year','value','type']], fc_sel[['year','value','type']]])

    # Format GDP as trillions
    if indicator == "gdp_current_usd":
        y_axis = alt.Y("value:Q", title="GDP (Trillions USD)", axis=alt.Axis(format="~s"))
    else:
        y_axis = alt.Y("value:Q", title=indicator)

    # Chart
    chart = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O"),
            y=y_axis,
            color="type:N",
            tooltip=["year", "value", "type"]
        )
        .interactive()
    )

    st.altair_chart(chart, width='stretch')

    # Table + download
    st.subheader(f"{chosen} — {country} — {indicator}")
    st.dataframe(fc_sel[['year','value']])

    csv = fc_sel[['year','value']].to_csv(index=False)
    st.download_button(
        "Download Scenario CSV",
        csv,
        file_name=f"{country}_{indicator}_{chosen.replace(' ', '_')}.csv"
    )

st.info("""
• Choose a scenario  
• Select a country + indicator  
• View historical + scenario trends  
• Download results  
""")
