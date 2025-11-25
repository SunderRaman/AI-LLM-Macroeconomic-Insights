
import pandas as pd
import os

PROC_PATH = os.path.join(os.getcwd(), 'data', 'processed')

def load_all_data():
    try:
        wb = pd.read_csv(os.path.join(PROC_PATH, 'wb_macro_clean.csv'))
    except Exception:
        wb = None
   
    # US macro: prefer us_macro_clean.csv, fallback to us_full_macro.csv
    us_full = None
    for fname in ['us_macro_clean.csv', 'us_full_macro.csv']:
        fpath = os.path.join(PROC_PATH, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            us_full = pd.read_csv(fpath)
            break
    
    try:
        forecasts = pd.read_csv(os.path.join(PROC_PATH, 'forecasts', 'forecasts_all.csv'))
    except Exception:
        forecasts = None
    return wb, us_full, forecasts
