import requests
import pandas as pd
import os

COUNTRIES = {
    "IN": "India",
    "US": "United States",
    "CN": "China",
    "GB": "United Kingdom",
    "DE": "Germany",
    "JP": "Japan"
}

INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
    "SP.POP.TOTL": "population"
}

START_YEAR = 2000
END_YEAR = 2024

RAW_DATA_PATH = "data/raw/worldbank"
os.makedirs(RAW_DATA_PATH, exist_ok=True)


def fetch_worldbank(country_code, indicator):
    """Fetch data from World Bank for ONE country + ONE indicator."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=2000"

    r = requests.get(url)
    r.raise_for_status()
    
    data_json = r.json()

    # If API returns an empty response
    if not isinstance(data_json, list) or len(data_json) < 2:
        return pd.DataFrame()

    data = data_json[1]  # actual records list

    df = pd.DataFrame(data)

    # Filter valid years
    df = df[pd.to_numeric(df["date"], errors="coerce").between(START_YEAR, END_YEAR)]

    df = df[["countryiso3code", "country", "date", "value"]]

    df.rename(columns={
        "countryiso3code": "country_code",
        "date": "year",
        "value": indicator
    }, inplace=True)

    return df


def run_worldbank_ingestion():
    print("\n=== 🌍 WORLD BANK INGESTION STARTED ===")

    for indicator, name in INDICATORS.items():
        print(f"\n--- Fetching {name} ({indicator}) ---")

        all_rows = []

        for ccode, cname in COUNTRIES.items():
            df = fetch_worldbank(ccode, indicator)

            if df.empty:
                print(f"⚠ No data for {indicator} in {ccode}")
                continue

            all_rows.append(df)

        if not all_rows:
            print(f"❌ Skipping {indicator}, no data found")
            continue

        final_df = pd.concat(all_rows, ignore_index=True)

        filepath = os.path.join(RAW_DATA_PATH, f"{name}.csv")
        final_df.to_csv(filepath, index=False)

        print(f"✅ Saved → {filepath}")

    print("\n=== ✔ World Bank ingestion completed ===")


if __name__ == "__main__":
    run_worldbank_ingestion()
