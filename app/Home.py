
import streamlit as st
from PIL import Image
from utils.load_data import load_all_data
import glob
import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



st.set_page_config(page_title='Macro AI Dashboard', layout='wide')

st.title('Global Macroeconomics Intelligence — Dashboard')
st.markdown('''This dashboard shows historical macro indicators and model forecasts (GDP & Inflation) for multiple countries.Use the pages to the left to navigate: Historical, Forecasts, Country Comparison, US Macro panel, and AI Insights.''')

# Get 3 random images from outputs/plots
image_paths = glob.glob("outputs/plots/*.png")[:3]

if len(image_paths) == 0:
    st.info("No plot images found in outputs/plots/")
else:
    cols = st.columns(3)
    for col, p in zip(cols, image_paths):
        try:
            img = Image.open(p)
            filename = p.split("/")[-1]
            col.image(img, caption=filename, width='stretch')
        except:
            col.write("Error loading image")

# quick data summary
wb, us_full, forecasts = load_all_data()
if wb is not None:
    cols = wb.columns.tolist()
    st.subheader('Data Snapshot')
    st.write('Processed World Bank rows:', wb.shape[0])
    st.write('Available countries:', wb['country'].unique().tolist())

st.sidebar.title('Navigation')
st.sidebar.info('Use the pages menu to go to different sections of the dashboard')
