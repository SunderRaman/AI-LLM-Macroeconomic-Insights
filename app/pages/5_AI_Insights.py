
import streamlit as st
from utils.load_data import load_all_data

st.title('AI Insights (placeholder)')
st.write('This page will later show LLM-generated summaries and explanations of forecasts.')

wb, us_full, forecasts = load_all_data()
if st.button('Generate Sample Insight (placeholder)'):
    st.info('LLM integration pending — will summarize forecasts here')
