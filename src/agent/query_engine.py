import pandas as pd
import os
import google.generativeai as genai
# from retriever import retrieve_context # pyright: ignore[reportMissingImports]

from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


DATA_PROC = "data/processed"
FORECAST_FILE = os.path.join(DATA_PROC, "forecasts", "forecasts_all.csv")
SCENARIO_FILE = os.path.join(DATA_PROC, "forecasts", "scenarios", "all_scenarios_combined.csv")
BACKTEST_FILE = os.path.join(DATA_PROC, "backtest_metrics_multi.csv")

baseline_df = pd.read_csv(FORECAST_FILE)

def load_country_data(country):
    """Load relevant data for a country from all sources."""
    data = {}

    try:
        fc = pd.read_csv(FORECAST_FILE)
        data["forecast"] = fc[fc["country"] == country]
    except:
        data["forecast"] = pd.DataFrame()

    try:
        scen = pd.read_csv(SCENARIO_FILE)
        data["scenario"] = scen[(scen["country"] == country) &
                                (scen["indicator"] == "gdp_current_usd")]
    except:
        data["scenario"] = pd.DataFrame()

    try:
        bt = pd.read_csv(BACKTEST_FILE)
        data["backtest"] = bt[bt["country"] == country]
    except:
        data["backtest"] = pd.DataFrame()

    return data


def summarize_facts(data, country):
    """Extract structured facts to constrain LLM output."""
    out = []

    # forecast info
    fc = data["forecast"]
    if not fc.empty:
        last = fc["forecast_mean"].iloc[-1]
        out.append(f"Latest forecast GDP: {last:.2f} trillion USD")

    # scenario divergence
    scen = data["scenario"]
    if not scen.empty:
        base_2030 = scen[(scen["year"] == 2030) &
                         (scen["scenario"] == "Baseline")]["scenario_forecast"].values
        trade_2030 = scen[(scen["year"] == 2030) &
                          (scen["scenario"].str.contains('Trade'))]["scenario_forecast"].values
        if len(base_2030) > 0 and len(trade_2030) > 0:
            diff = trade_2030[0] - base_2030[0]
            out.append(f"Trade impact around 2030: {diff:.3f} trillion USD")

    # backtesting confidence
    bt = data["backtest"]
    if not bt.empty:
        mape1 = bt[bt["horizon"]==1]["mape_pct"].mean()
        mape3 = bt[bt["horizon"]==3]["mape_pct"].mean()
        out.append(f"Forecast MAPE 1 year: {mape1:.2f}%")
        out.append(f"Forecast MAPE 3 year: {mape3:.2f}%")

    return "\n".join(out)


def answer_query(country, question):
    """Main entry point for Option-B agentic QA."""
    data = load_country_data(country)
    # context = retrieve_context(country)
    #facts = summarize_facts(data, country)
    facts_df = baseline_df[
        (baseline_df["country"] == country) &
        (baseline_df["indicator"] == "gdp_current_usd")
    ][["year","forecast_mean"]]

    # Load uncertainty
    bm_multi = pd.read_csv(
        "data/processed/backtesting/backtest_metrics_multi.csv"
    )

    uncert_df = bm_multi[
        (bm_multi["country"] == country) &
        (bm_multi["indicator"] == "gdp_current_usd")
    ][["horizon","mape_pct"]]

    # Build facts block
    facts = (
        "BASELINE FORECAST:\n" +
        facts_df.to_string(index=False) +
        "\n\nFORECAST UNCERTAINTY (MAPE % by horizon):\n" +
        uncert_df.to_string(index=False)
    )

    prompt = f"""
    You are an economist answering a question about {country}'s GDP outlook.

    GROUND TRUTH DATA:
    {facts}

    DATA RULES (STRICT):
    - GDP levels come ONLY from 'forecast_mean'
    - MAPE values represent forecast uncertainty
    - GDP cannot be negative in this dataset

    TASK:
    Provide a 3-5 sentence professional interpretation that ALWAYS includes:
    - the baseline GDP trend
    - how forecast uncertainty changes over time
    - what increasing uncertainty implies for volatility

    Do NOT output bullet points.
    USER QUESTION:
    {question}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()
