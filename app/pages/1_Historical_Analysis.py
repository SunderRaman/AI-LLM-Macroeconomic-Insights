
import streamlit as st
import altair as alt
from utils.load_data import load_all_data

st.title('Historical Analysis')
wb, us_full, forecasts = load_all_data()

country = st.selectbox('Country', wb['country'].unique().tolist())
indicator = st.selectbox('Indicator', ['gdp_current_usd','gdp_growth_pct','inflation_cpi_pct','unemployment_pct','exports_pct_gdp','imports_pct_gdp','population'])

series = wb[wb['country']==country][['year', indicator]].dropna()
if series.empty:
    st.warning('No data available for this selection')
else:
    # If GDP, format in trillions for display
    if indicator == "gdp_current_usd":
        y_encoding = alt.Y(
            f'{indicator}:Q',
            title="GDP (Trillions USD)",
            axis=alt.Axis(format="~s")
        )
    else:
        y_encoding = alt.Y(f'{indicator}:Q', title=indicator)

    chart = alt.Chart(series).mark_line().encode(
        x='year:O',
        y=y_encoding,
        tooltip=['year', indicator]
    ).interactive()
    st.altair_chart(chart, width='stretch')
