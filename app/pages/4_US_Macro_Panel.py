import streamlit as st
import altair as alt
from utils.load_data import load_all_data

st.title('US Macro Panel')

wb, us_full, forecasts = load_all_data()

if us_full is None:
    st.error("US macro data not found.")
    st.stop()

st.write("Showing indicators from us_macro_clean.csv")

# --- Identify columns dynamically ---
possible_fed_cols = [c for c in us_full.columns if "fed" in c.lower() or "fund" in c.lower()]
possible_spread_cols = [c for c in us_full.columns if "spread" in c.lower() or "10" in c and "2" in c]

fed_col = possible_fed_cols[0] if possible_fed_cols else None
spread_col = possible_spread_cols[0] if possible_spread_cols else None

# --- Fed Funds plot ---
st.subheader('Fed Funds Rate')

if fed_col is None:
    st.warning("Could not find a fed funds rate column in us_full_macro.csv")
else:
    chart = alt.Chart(us_full).mark_line().encode(
        x='year:O',
        y=alt.Y(f'{fed_col}:Q', title=fed_col),
        tooltip=['year', fed_col]
    ).interactive()

    st.altair_chart(chart, width='stretch')


# --- Yield Spread plot ---
st.subheader('Yield Spread (10y - 2y)')

if spread_col is None:
    st.warning("Could not find a yield spread column in us_full_macro.csv")
else:
    chart2 = alt.Chart(us_full).mark_line().encode(
        x='year:O',
        y=alt.Y(f'{spread_col}:Q', title=spread_col),
        tooltip=['year', spread_col]
    ).interactive()

    st.altair_chart(chart2, width='stretch')
