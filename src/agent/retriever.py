import pandas as pd
import os
import re
from src.agent.pdf_loader import extract_pdf_text  # pyright: ignore[reportMissingImports]

BASE_PATH = "data/processed"

def retrieve_context(question: str, country: str):
    q = question.lower()
    ctx = []

    # ----------------------------------------
    # Detect "comparison" questions
    # ----------------------------------------
    compare_mode = any(kw in q for kw in [
        "which country",
        "highest",
        "lowest",
        "most",
        "least",
        "compare",
        "across countries",
        "countries"
    ])

    # ----------------------------------------
    # MULTI-COUNTRY MODE
    # ----------------------------------------
    if compare_mode:
        try:
            bt = pd.read_csv(f"{BASE_PATH}/backtesting/backtest_metrics_multi.csv")

            # Only GDP forecast uncertainty horizon=3
            df3 = bt[(bt.indicator=="gdp_current_usd") & (bt.horizon==3)][
                ["country","mape_pct"]
            ].sort_values("mape_pct", ascending=False)

            ctx.append("LONG-TERM FORECAST UNCERTAINTY BY COUNTRY (MAPE horizon 3):\n" +
                       df3.to_string(index=False))

            return "\n\n".join(ctx)[:2000]

        except Exception as e:
            ctx.append(f"(comparison mode failed: {e})")

    # ----------------------------------------
    # SINGLE COUNTRY MODE (your original logic)
    # ----------------------------------------
    # Baseline forecasts
    try:
        fc = pd.read_csv(f"{BASE_PATH}/forecasts/forecasts_all.csv")
        fc_country = fc[(fc.country == country) & (fc.indicator=="gdp_current_usd")]
        ctx.append("BASELINE FORECAST:\n" + fc_country.to_string(index=False))
    except:
        pass

    # Uncertainty
    try:
        bt = pd.read_csv(f"{BASE_PATH}/backtesting/backtest_metrics_multi.csv")
        bt_country = bt[(bt.country == country) & (bt.indicator=="gdp_current_usd")]
        ctx.append("\nFORECAST UNCERTAINTY:\n" + bt_country.to_string(index=False))
    except:
        pass

    # Scenarios
    try:
        scen = pd.read_csv(f"{BASE_PATH}/forecasts/scenarios/all_scenarios_combined.csv")
        scen_country = scen[(scen.country == country) & (scen.indicator=="gdp_current_usd")]
        # Pivot so scenarios are side-by-side for comparison
        scen_pivot = scen_country.pivot(
            index="year",
            columns="scenario",
            values="scenario_forecast"
        ).reset_index()

        ctx.append("\nSCENARIOS:\n" + scen_pivot.head(10).to_string(index=False))
    except Exception as e:
        ctx.append(f"\n(Scenario load failed: {e})")

    # PDF text
    try:
        pdf_text = extract_pdf_text(f"outputs/reports/{country}_GDP_Report.pdf")
        ctx.append("\nPDF REPORT TEXT:\n" + pdf_text)
    except:
        pass

    max_len = 6000 if "scenario" in q or "downside" in q or "upside" in q else 2000
    return "\n\n".join(ctx)[:max_len]
