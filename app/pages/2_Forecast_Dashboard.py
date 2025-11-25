
import streamlit as st
import pandas as pd
import altair as alt
from utils.load_data import load_all_data

st.title('Forecast Dashboard')
wb, us_full, forecasts = load_all_data()

country = st.selectbox('Country (Forecast)', forecasts['country'].unique().tolist())
indicator = st.selectbox('Indicator (Forecast)', forecasts['indicator'].unique().tolist())

sel = forecasts[(forecasts['country']==country)&(forecasts['indicator']==indicator)]
if sel.empty:
    st.warning('No forecast data')
else:
    hist = wb[(wb['country']==country)][['year', indicator]].dropna()
    fc = sel
    combined = pd.concat([hist.rename(columns={indicator:'value'}).assign(type='historical')[['year','value','type']],
                          fc.rename(columns={'forecast_mean':'value'}).assign(type='forecast')[['year','value','type']]],
                         ignore_index=True)
    if indicator == "gdp_current_usd":
        y_enc = alt.Y('value:Q', title="GDP (Trillions USD)", axis=alt.Axis(format="~s"))
    else:
        y_enc = alt.Y('value:Q', title=indicator)

    chart = alt.Chart(combined).mark_line().encode(
        x='year:O',
        y=y_enc,
        color='type:N',
        tooltip=['year','value','type']
    ).interactive()
    st.altair_chart(chart,width='stretch')
    st.download_button('Download Forecast CSV', data=sel.to_csv(index=False), file_name=f'{country}_{indicator}_forecast.csv')
