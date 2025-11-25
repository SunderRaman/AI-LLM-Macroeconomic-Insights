# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
PROC_PATH = "data/processed"
PLOT_PATH = "outputs/plots"
os.makedirs(PLOT_PATH, exist_ok=True)

# Load Datasets
wb = pd.read_csv(os.path.join(PROC_PATH, "wb_macro_clean.csv"))
us_macro = pd.read_csv(os.path.join(PROC_PATH, "us_macro_clean.csv"))
us_full = pd.read_csv(os.path.join(PROC_PATH, "us_full_macro.csv"))

# Countries to plot
countries = ["India", "United States", "China", "United Kingdom", "Germany", "Japan"]

sns.set(style="whitegrid")

COUNTRY_COLORS = {
    "India": "#1f77b4",           # Blue
    "United States": "#d62728",   # Red
    "China": "#2ca02c",           # Green
    "United Kingdom": "#9467bd",  # Purple
    "Germany": "#8c564b",         # Brown/Gold
    "Japan": "#17becf"            # Teal/Cyan
}

# ------------------------------------------------------------
# Helper function to save plots
# ------------------------------------------------------------
def save_plot(fig, name):
    filepath = os.path.join(PLOT_PATH, name)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"📁 Saved → {filepath}")


# ------------------------------------------------------------
# 1. GDP Trend: 2000–2024
# ------------------------------------------------------------
def plot_gdp_trend():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        subset = wb[wb["country"] == c]
        ax.plot(subset["year"], subset["gdp_current_usd"], label=c, color=COUNTRY_COLORS[c])

    ax.set_title("GDP (Current USD) — 2000 to 2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (USD)")
    ax.legend()
    save_plot(fig, "gdp_trend.png")


# ------------------------------------------------------------
# 2. GDP Growth (%) Comparison
# ------------------------------------------------------------
def plot_gdp_growth():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        ss = wb[wb["country"] == c]
        ax.plot(ss["year"], ss["gdp_growth_pct"], label=c, color=COUNTRY_COLORS[c])

    ax.set_title("GDP Growth (%) — 2000 to 2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth (%)")
    ax.legend()
    save_plot(fig, "gdp_growth.png")


# ------------------------------------------------------------
# 3. Inflation Comparison (CPI %)
# ------------------------------------------------------------
def plot_inflation():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        ss = wb[wb["country"] == c]
        ax.plot(ss["year"], ss["inflation_cpi_pct"], label=c, color=COUNTRY_COLORS[c])

    ax.set_title("Inflation (CPI %) — 2000 to 2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation (%)")
    ax.legend()
    save_plot(fig, "inflation.png")


# ------------------------------------------------------------
# 4. Unemployment % Comparison
# ------------------------------------------------------------
def plot_unemployment():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        ss = wb[wb["country"] == c]
        ax.plot(ss["year"], ss["unemployment_pct"], label=c, color=COUNTRY_COLORS[c])

    ax.set_title("Unemployment (%) — 2000 to 2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment (%)")
    ax.legend()
    save_plot(fig, "unemployment.png")


# ------------------------------------------------------------
# 5. Exports vs Imports (% of GDP)
# ------------------------------------------------------------
def plot_exports_imports():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        ss = wb[wb["country"] == c]
        ax.plot(
            ss["year"], 
            ss["exports_pct_gdp"],
            label=f"{c} Exports",
            color=COUNTRY_COLORS[c],
            linewidth=2.0
        )

        ax.plot(
            ss["year"], 
            ss["imports_pct_gdp"],
            label=f"{c} Imports",
            color=COUNTRY_COLORS[c],
            linestyle="--",
            linewidth=2.0
        )

    ax.set_title("Exports vs Imports (% of GDP)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of GDP")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, "exports_imports.png")


# ------------------------------------------------------------
# 6. Population Trend
# ------------------------------------------------------------
def plot_population():
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in countries:
        ss = wb[wb["country"] == c]
        ax.plot(ss["year"], ss["population"], label=c, color=COUNTRY_COLORS[c])

    ax.set_title("Population — 2000 to 2024")
    ax.set_xlabel("Year")
    ax.set_ylabel("People")
    ax.legend()
    save_plot(fig, "population.png")


# ------------------------------------------------------------
# 7. US Federal Funds Rate
# ------------------------------------------------------------
def plot_fed_funds():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(us_macro["year"], us_macro["fed_funds_rate"], color="blue")

    ax.set_title("Federal Funds Rate (US)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Interest Rate (%)")
    save_plot(fig, "fed_funds_rate.png")


# ------------------------------------------------------------
# 8. Yield Curve (10Y - 2Y)
# ------------------------------------------------------------
def plot_yield_curve():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(us_macro["year"], us_macro["yield_spread_10y_2y"], color="red")

    ax.axhline(0, color="black", linestyle="--")
    ax.set_title("Yield Curve Spread (10Y - 2Y)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Spread (%)")
    save_plot(fig, "yield_curve_spread.png")


# ------------------------------------------------------------
# 9. Recession Indicator
# ------------------------------------------------------------
def plot_recession():
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(us_macro["year"], us_macro["recession_indicator"], color="purple")

    ax.set_title("US Recession Indicator (NBER)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recession (0/1)")
    save_plot(fig, "recession_indicator.png")


# ------------------------------------------------------------
# MAIN RUN FUNCTION
# ------------------------------------------------------------
def run_eda():
    print("\n=== 📊 Running EDA ===\n")

    plot_gdp_trend()
    plot_gdp_growth()
    plot_inflation()
    plot_unemployment()
    plot_exports_imports()
    plot_population()
    plot_fed_funds()
    plot_yield_curve()
    plot_recession()

    print("\n=== ✔ EDA Complete — All plots saved ===")


if __name__ == "__main__":
    run_eda()
