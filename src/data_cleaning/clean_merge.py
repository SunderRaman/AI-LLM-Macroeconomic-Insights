# clean_merge.py

import pandas as pd
import os
import ast

RAW_WB = "data/raw/worldbank"
RAW_FRED = "data/raw/fred"
PROC_PATH = "data/processed"
os.makedirs(PROC_PATH, exist_ok=True)

START_YEAR = 2000
END_YEAR = 2024

# Countries
COUNTRIES = {
    'IN': 'India',
    'US': 'United States',
    'CN': 'China',
    'GB': 'United Kingdom',
    'DE': 'Germany',
    'JP': 'Japan'
}

# ------------------------------------------
# 1. LOAD & MERGE WORLD BANK INDICATORS
# ------------------------------------------

def load_worldbank():
    print("\n📥 Loading World Bank CSV Files...")
    
    wb_files = {
        "gdp_current_usd": "gdp_current_usd.csv",
        "gdp_growth_pct": "gdp_growth_pct.csv",
        "gdp_per_capita": "gdp_per_capita.csv",
        "inflation_cpi_pct": "inflation_cpi_pct.csv",
        "unemployment_pct": "unemployment_pct.csv",
        "exports_pct_gdp": "exports_pct_gdp.csv",
        "imports_pct_gdp": "imports_pct_gdp.csv",
        "population": "population.csv"
    }

    dfs = []

    for col, fname in wb_files.items():
        path = os.path.join(RAW_WB, fname)
        df = pd.read_csv(path)

        # Fix country field (parse dict stored as string)
        df["country"] = df["country"].apply(
            lambda x: ast.literal_eval(x)['value']
            if isinstance(x, str) and x.startswith("{")
            else x
        )

        df["year"] = df["year"].astype(int)

        # Rename indicator columns
        rename_map = {
            "NY.GDP.MKTP.CD": "gdp_current_usd",
            "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
            "NY.GDP.PCAP.CD": "gdp_per_capita",
            "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
            "SL.UEM.TOTL.ZS": "unemployment_pct",
            "NE.EXP.GNFS.ZS": "exports_pct_gdp",
            "NE.IMP.GNFS.ZS": "imports_pct_gdp",
            "SP.POP.TOTL": "population"
        }

        df.rename(columns=rename_map, inplace=True)

        dfs.append(df)

    # Start merging on (country_code, year)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df, on=["country_code", "country", "year"], how="outer"
        )

    # Filter years
    merged = merged[(merged["year"] >= START_YEAR) & (merged["year"] <= END_YEAR)]

    save_path = os.path.join(PROC_PATH, "wb_macro_clean.csv")
    merged.to_csv(save_path, index=False)
    
    print(f"✅ Saved World Bank Cleaned → {save_path}")
    return merged


# ------------------------------------------
# 2. CLEAN FRED DATA
# ------------------------------------------

def load_fred():
    print("\n📥 Loading FRED CSV Files...")
    
    fred_files = {
        "fed_funds_rate": "fed_funds_rate.csv",
        "cpi_us": "cpi_us.csv",
        "m2_money_supply": "m2_money_supply.csv",
        "yield_10yr": "yield_10yr.csv",
        "yield_2yr": "yield_2yr.csv",
        "yield_spread_10y_2y": "yield_spread_10y_2y.csv",
        "recession_indicator": "recession_indicator.csv"
    }

    dfs = []

    for col, fname in fred_files.items():
        path = os.path.join(RAW_FRED, fname)
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        df = df.groupby("year")[col].mean().reset_index()
        
        dfs.append(df)

    # Merge all FRED datasets on year
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="year", how="outer")

    # Filter to same year range
    merged = merged[(merged["year"] >= START_YEAR) & (merged["year"] <= END_YEAR)]

    save_path = os.path.join(PROC_PATH, "us_macro_clean.csv")
    merged.to_csv(save_path, index=False)

    print(f"✅ Saved US FRED Cleaned → {save_path}")
    return merged


# ------------------------------------------
# 3. MERGE WORLD BANK + FRED FOR USA
# ------------------------------------------

def merge_us(wb_df, fred_df):
    print("\n🔗 Merging World Bank + FRED for USA...")

    us_wb = wb_df[wb_df["country_code"] == "US"]

    merged = us_wb.merge(fred_df, on="year", how="left")

    save_path = os.path.join(PROC_PATH, "us_full_macro.csv")
    merged.to_csv(save_path, index=False)

    print(f"✅ Saved US Combined Macro → {save_path}")


# ------------------------------------------
# MAIN PIPELINE
# ------------------------------------------

def run_clean_merge():
    print("\n=== 🧹 MODULE 2: Cleaning & Merge Started ===")

    wb_df = load_worldbank()
    fred_df = load_fred()
    merge_us(wb_df, fred_df)

    print("\n=== ✔ MODULE 2 COMPLETED ===")


if __name__ == "__main__":
    run_clean_merge()
