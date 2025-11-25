# fred_api.py

from fredapi import Fred
import pandas as pd
import os

# -------------------------------------------------
# Your FRED API Key (Masked version you provided)
# Replace with actual key in your local file
# -------------------------------------------------
FRED_API_KEY = "235cce698083bd5e87f0ea0ff1e3b5de"

fred = Fred(api_key=FRED_API_KEY)

RAW_DATA_PATH = "data/raw/fred"
os.makedirs(RAW_DATA_PATH, exist_ok=True)

# -------------------------------------------------
# Key US Macroeconomic Indicators
# -------------------------------------------------
FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",          # Federal Funds Rate
    "CPIAUCSL": "cpi_us",                  # US CPI (Index)
    "M2SL": "m2_money_supply",             # M2 Money Supply
    "DGS10": "yield_10yr",                 # 10-Year Treasury Yield
    "DGS2": "yield_2yr",                   # 2-Year Treasury Yield
    "T10Y2Y": "yield_spread_10y_2y",       # Yield Curve Spread
    "USREC": "recession_indicator"         # NBER Recession Indicator (0/1)
}


# -------------------------------------------------
# Function to fetch & save FRED series
# -------------------------------------------------
def fetch_fred_series(series_code, file_name):
    print(f"\n📥 Fetching: {file_name} ({series_code})")

    try:
        series = fred.get_series(series_code)

        df = pd.DataFrame({
            "date": series.index,
            file_name: series.values
        })

        # Save CSV
        filepath = os.path.join(RAW_DATA_PATH, f"{file_name}.csv")
        df.to_csv(filepath, index=False)

        print(f"✅ Saved → {filepath}")

        return df

    except Exception as e:
        print(f"⚠ Error fetching {series_code}: {e}")
        return pd.DataFrame()


# -------------------------------------------------
# Main runner
# -------------------------------------------------
def run_fred_ingestion():
    print("\n=== 🇺🇸 FRED INGESTION STARTED ===")

    for series_code, file_name in FRED_SERIES.items():
        fetch_fred_series(series_code, file_name)

    print("\n=== ✔ COMPLETED: FRED INGESTION ===")


if __name__ == "__main__":
    run_fred_ingestion()
