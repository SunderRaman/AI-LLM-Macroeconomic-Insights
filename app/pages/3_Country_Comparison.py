
import streamlit as st
import altair as alt
from utils.load_data import load_all_data

st.title('Country Comparison')
wb, us_full, forecasts = load_all_data()

indicator = st.selectbox('Indicator (Compare)', ['gdp_current_usd','gdp_growth_pct','inflation_cpi_pct'])

plot_df = wb.pivot(index='year', columns='country', values=indicator).reset_index()

if indicator == "gdp_current_usd":
    y_axis = alt.Y("value:Q", title="GDP (Trillions USD)", axis=alt.Axis(format="~s"))
else:
    y_axis = alt.Y("value:Q", title=indicator)

chart = (
    alt.Chart(plot_df)
    .transform_fold(
        [c for c in plot_df.columns if c != 'year'],
        as_=['country', 'value']
    )
    .mark_line()
    .encode(
        x=alt.X('year:O'),
        y=y_axis,
        color=alt.Color('country:N', title='Country'),
        tooltip=['year:O', 'country:N', 'value:Q']
    )
    .interactive()
)
st.altair_chart(chart, width='stretch')
