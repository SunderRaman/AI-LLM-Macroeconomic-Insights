# src/report/pdf_report.py

import os
import io
from textwrap import wrap

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime


DATA_PROC = os.path.join("data", "processed")
FORECASTS_ALL = os.path.join(DATA_PROC, "forecasts", "forecasts_all.csv")
BT_HORIZON = os.path.join(DATA_PROC, "backtesting", "backtest_metrics_by_horizon.csv")
SCENARIOS_ALL = os.path.join(DATA_PROC, "forecasts", "scenarios", "all_scenarios_combined.csv")
WB_CLEAN = os.path.join(DATA_PROC, "wb_macro_clean.csv")

REPORT_DIR = os.path.join("outputs", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

DEFAULT_GDP_COUNTRIES = [
    "India",
    "China",
    "United States",
    "United Kingdom",
    "Germany",
    "Japan",
]

def _load_gemini_model():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set (in env or .env)")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={"response_mime_type": "text/plain"},
    )
    return model


def _build_forecast_band(country: str):
    """Return:
    hist_df (year, value_trn),
    fc_df (year, value_trn, lower_trn, upper_trn),
    bt_summary (for GDP & horizons)
    """
    wb = pd.read_csv(WB_CLEAN)
    fc = pd.read_csv(FORECASTS_ALL)
    bt = pd.read_csv(BT_HORIZON)

    wb = wb[wb["country"] == country].copy()
    fc = fc[(fc["country"] == country) & (fc["indicator"] == "gdp_current_usd")].copy()
    bt = bt[(bt["country"] == country) & (bt["indicator"] == "gdp_current_usd")].copy()

    wb["year"] = wb["year"].astype(int)
    fc["year"] = fc["year"].astype(int)
    bt["horizon"] = bt["horizon"].astype(int)

    # historical GDP -> trillions
    hist = wb[["year", "gdp_current_usd"]].dropna().copy()
    hist = hist.sort_values("year")
    hist["gdp_trn"] = hist["gdp_current_usd"] / 1_000_000_000_000.0

    last_hist_year = hist["year"].max()

    # forecast_mean column name -> value
    if "forecast_mean" in fc.columns:
        fc["value"] = fc["forecast_mean"]
    elif "value" not in fc.columns:
        raise ValueError("forecasts_all.csv missing forecast_mean/value column for GDP")

    fc = fc.sort_values("year").copy()

    # build horizon -> MAPE dict
    if bt.empty:
        # no backtest info; no band
        fc["gdp_trn"] = fc["value"] / 1_000_000_000_000.0
        fc["lower_trn"] = fc["gdp_trn"]
        fc["upper_trn"] = fc["gdp_trn"]
        fc["mape_pct"] = 0.0
        return hist[["year", "gdp_trn"]], fc[["year", "gdp_trn", "lower_trn", "upper_trn", "mape_pct"]], bt

    h2m = {int(r["horizon"]): float(r["mape_pct"]) for _, r in bt.iterrows()}
    max_h = max(h2m.keys())

    def pick_mape(year):
        h = year - last_hist_year
        if h <= 0:
            return 0.0
        h_use = h if h in h2m else max_h
        return h2m.get(h_use, 0.0)

    fc["mape_pct"] = fc["year"].apply(pick_mape)
    fc["gdp_trn"] = fc["value"] / 1_000_000_000_000.0
    fc["lower_trn"] = fc["gdp_trn"] * (1 - fc["mape_pct"] / 100.0)
    fc["upper_trn"] = fc["gdp_trn"] * (1 + fc["mape_pct"] / 100.0)

    return hist[["year", "gdp_trn"]], fc[["year", "gdp_trn", "lower_trn", "upper_trn", "mape_pct"]], bt


def _build_scenario_df(country: str):
    """Return a dataframe with columns:
       year, baseline_trn, trade_trn, optimistic_trn (if available)
    If scenarios file missing, return None.
    """
    if not os.path.exists(SCENARIOS_ALL):
        return None

    scen = pd.read_csv(SCENARIOS_ALL)
    # print(scen[scen["country"]=="India"][["year","scenario","scenario_forecast"]].head(15))
    if "scenario" not in scen.columns:
        # user may have different structure; bail gracefully
        return None

    scen = scen[
        (scen["country"] == country)
        & (scen["indicator"] == "gdp_current_usd")
    ].copy()
    if scen.empty:
        return None

    # scenario_forecast assumed to be in trillions already; if not, user can adjust
    val_col = "scenario_forecast"
    if "scenario_forecast" not in scen.columns:
        raise ValueError("scenarios file missing 'scenario_forecast' column")

    # pivot to columns by scenario name
    pivot = scen.pivot_table(
        index="year", columns="scenario", values=val_col, aggfunc="mean"
    ).reset_index()

    # make column names friendlier
    pivot.columns = [str(c) for c in pivot.columns]
    pivot["year"] = pivot["year"].astype(int)

    # we don't know exact scenario names; keep all, user can see in legend
    return pivot


def _plot_forecast_band_png(country: str):
    hist, fc, bt = _build_forecast_band(country)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["year"], hist["gdp_trn"], label="Historical", color="black", linewidth=1)
    ax.plot(fc["year"], fc["gdp_trn"], label="Forecast", color="navy", linewidth=1.5)
    ax.fill_between(fc["year"], fc["lower_trn"], fc["upper_trn"],
                    color="skyblue", alpha=0.3, label="Uncertainty band")

    ax.set_title(f"{country} — GDP Forecast with Uncertainty")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (Trillion USD)")
    ax.grid(alpha=0.2)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf, hist, fc, bt


def _plot_scenario_png(country: str):
    scen_df = _build_scenario_df(country)
    if not os.path.exists(SCENARIOS_ALL):
        return None

    scen = pd.read_csv(SCENARIOS_ALL)

    base = scen_df["Baseline"]

    for col in scen_df.columns:
        if col == "year" or col == "Baseline":
            continue
        scen_df[col + "_pct"] = (scen_df[col] - base) / base * 100.0

    fig, ax = plt.subplots(figsize=(4,2))

    for col in scen_df.columns:
        if col.endswith("_pct"):
            ax.plot(
                scen_df["year"],
                scen_df[col],
                label=col.replace("_pct",""),
                linewidth=1.5
            )

    ax.set_title(f"{country} — Scenario Impact vs Baseline (%)")
    ax.set_ylabel("% difference from baseline")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf, scen_df
    
def _draw_title_page(c, country: str, start_year: int, end_year: int):
    """Draws a formal title page on the given canvas."""
    width, height = A4
    margin = 50

    # Background / base
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, stroke=0, fill=1)

    # Main title
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 22)
    title = "Macroeconomic Forecast & Scenario Analysis Report"
    tw = c.stringWidth(title, "Helvetica-Bold", 22)
    c.drawString((width - tw) / 2, height - 150, title)

    # Country + horizon
    c.setFont("Helvetica-Bold", 18)
    subtitle = f"{country} — {start_year} to {end_year}"
    sw = c.stringWidth(subtitle, "Helvetica-Bold", 18)
    c.drawString((width - sw) / 2, height - 190, subtitle)

    # Prepared for / by
    c.setFont("Helvetica", 11)
    line1 = "Prepared for: Portfolio Use / Demonstration"
    line2 = "Prepared by: Sunder Raman V"
    line3 = f"Generated on: {datetime.now().strftime('%d %b %Y')}"
    line4 = "Tools: Python • Machine Learning • Gemini LLM • Streamlit"

    c.drawString(margin, height - 240, line1)
    c.drawString(margin, height - 260, line2)
    c.drawString(margin, height - 280, line3)
    c.drawString(margin, height - 300, line4)

    # Optional bottom note
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(
        margin,
        margin,
        "This report is auto-generated from a macroeconomics + AI forecasting pipeline.",
    )

def _build_llm_summary(model, country: str, fc_band_df: pd.DataFrame, bt_df: pd.DataFrame):
    # use last 3–4 forecast years for compact table in prompt
    fc_tail = fc_band_df.sort_values("year").tail(4)[["year", "gdp_trn", "lower_trn", "upper_trn", "mape_pct"]]

    bt_small = (
        bt_df.sort_values("horizon")[["horizon", "mape_pct"]]
        if not bt_df.empty
        else pd.DataFrame(columns=["horizon", "mape_pct"])
    )

    prompt = f"""
You are an expert macroeconomic analyst writing a concise professional brief.

Country: {country}
Indicator: Nominal GDP (Trillion USD)

Forecast with uncertainty (last few years, in trillions):
{fc_tail.to_string(index=False)}

Backtesting accuracy (MAPE by horizon, in %):
{bt_small.to_string(index=False)}

Instructions:
- Write 1 short paragraph (4–6 sentences).
- Use **plain text** only (no Markdown, no bullets, no italics).
- Do not break words or numbers across lines.
- Include:
  - the overall growth direction and approximate level by the last forecast year;
  - how wide the uncertainty band is and how it changes with horizon;
  - whether the model is more reliable in the short term or medium term;
  - 1–2 key upside/downside risks (e.g., trade, global growth, domestic policy, inflation).
- Tone: professional, clear, non-technical.
"""

    resp = model.generate_content(prompt)
    return resp.text.strip()


def _build_llm_risks(model, country: str):
    prompt = f"""
You are an economist. Give 3–4 concise bullet-style risk/driver statements 
for {country}'s GDP outlook over the next 5–10 years.

Output rules:
- Plain text.
- Each line should start with "- ".
- No Markdown formatting, no italics, no bold.
- Keep each bullet short (max 20 words).
"""
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    # basic cleanup: keep only lines starting with '-'
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("-")]
    return lines[:4] if lines else [text]


def _draw_wrapped_text(c, text, x, y, max_width, line_height, font_name="Helvetica", font_size=10):
    c.setFont(font_name, font_size)
    # rough width approximation using 0.5 * font_size per char
    max_chars = int(max_width / (0.5 * font_size))
    wrapped = []
    for p in text.split("\n"):
        wrapped.extend(wrap(p, max_chars))
    for line in wrapped:
        c.drawString(x, y, line)
        y -= line_height
    return y


def generate_gdp_report(country: str = "India", output_path: str | None = None):
    """
    Generate a multi-page GDP PDF report for the given country.
    - Uses Gemini 2.5 Flash for narrative sections.
    - Uses forecasting + backtesting to build uncertainty band.
    - Uses scenarios file (if present) for comparison chart.

    Output: outputs/reports/<Country>_GDP_Report.pdf
    """
    if output_path is None:
        safe_country = country.replace(" ", "_")
        output_path = os.path.join(REPORT_DIR, f"{safe_country}_GDP_Report.pdf")

    # 1. Build charts & data
    forecast_buf, hist_df, fc_band_df, bt_df = _plot_forecast_band_png(country)
    scenario_result = _plot_scenario_png(country)
    scen_buf, scen_df = (scenario_result if scenario_result is not None else (None, None))

    # Determine horizon for title page
    try:
        start_year = int(hist_df["year"].min())
        end_year = int(fc_band_df["year"].max())
    except Exception:
        start_year = 2000
        end_year = 2034

    # 2. LLM summaries
    model = _load_gemini_model()
    summary_text = _build_llm_summary(model, country, fc_band_df, bt_df)
    risks_list = _build_llm_risks(model, country)

    # 3. Some quick backtest metrics to print numerically
    bt_small = (
        bt_df.sort_values("horizon")[["horizon", "mape_pct"]].dropna()
        if not bt_df.empty
        else pd.DataFrame(columns=["horizon", "mape_pct"])
    )

    # 4. Create PDF
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 40

    # === TITLE PAGE ===
    _draw_title_page(c, country, start_year, end_year)
    c.showPage()

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, height - margin, f"{country} — GDP Outlook (Baseline & Scenarios)")

    # Executive summary
    y = height - margin - 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Executive Summary")
    y -= 18
    y = _draw_wrapped_text(c, summary_text, margin, y, width - 2 * margin, line_height=13)

    # Backtesting metrics (small table)
    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Backtesting Accuracy (MAPE by Horizon)")
    y -= 14
    c.setFont("Helvetica", 9)
    if bt_small.empty:
        c.drawString(margin, y, "No backtesting metrics available.")
        y -= 14
    else:
        for _, row in bt_small.iterrows():
            c.drawString(
                margin,
                y,
                f"Horizon {int(row['horizon'])} year(s): {row['mape_pct']:.1f}%",
            )
            y -= 12

    # Risks / Drivers
    y -= 4
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Key Drivers & Risks")
    y -= 14
    c.setFont("Helvetica", 9)
    for r in risks_list:
        c.drawString(margin, y, r)
        y -= 12

    # Reserve lower half for charts
    chart_top = height * 0.52
    chart_width = width / 2 - 1.5 * margin
    chart_height = height * 0.28

    # Forecast chart
    if forecast_buf is not None:
        img1 = ImageReader(forecast_buf)
        c.drawImage(
            img1,
            margin,
            chart_top - chart_height,
            width=chart_width,
            height=chart_height,
            preserveAspectRatio=True,
            mask="auto",
        )

    # Scenario chart
    if scen_buf is not None:
        img2 = ImageReader(scen_buf)
        c.drawImage(
            img2,
            margin + chart_width + margin / 2,
            chart_top - chart_height,
            width=chart_width,
            height=chart_height,
            preserveAspectRatio=True,
            mask="auto",
        )

    # === LLM Scenario Interpretation ===
    if scen_buf is not None and scen_df is not None:
        summary_text = generate_gemini_summary(country, scen_df)

        # starting text position just below scenario chart
        text_y = chart_top - chart_height - 20

        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.black)
        c.drawString(
            margin + chart_width + margin / 2,
            text_y,
            "Scenario Interpretation:"
        )

        c.setFont("Helvetica", 8)
        text_y -= 12

        # wrap text manually to fit column width
        max_width = chart_width - 10
        words = summary_text.split()
        line = ""

        for w in words:
            test_line = line + w + " "
            if c.stringWidth(test_line, "Helvetica", 8) < max_width:
                line = test_line
            else:
                c.drawString(
                    margin + chart_width + margin / 2,
                    text_y,
                    line.strip()
                )
                text_y -= 10
                line = w + " "

        # draw last line
        if line:
            c.drawString(
                margin + chart_width + margin / 2,
                text_y,
                line.strip()
            )

        text_y -= 14

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawRightString(
        width - margin,
        margin / 2,
        "Generated by Macro AI Project (Python, ML, Gemini LLM)",
    )

    c.showPage()
    c.save()

    print(f"Saved GDP report: {output_path}")
    return output_path

def generate_gemini_summary(country, scen_df):
    """
    Generates a 3–5 sentence executive summary of scenario behavior.
    Uses percentage deviation scenario_df.
    """

    # Build a compact text table for the model
    sample = scen_df.tail(5).to_string(index=False)

    prompt = f"""
You are an economist writing a concise professional macroeconomic insight
for a portfolio report.

The following table shows GDP scenario divergence from baseline for {country},
expressed as % difference from baseline:

{sample}

Write a 3–5 sentence executive summary focusing on:

- whether GDP is sensitive to trade or recession shocks
- magnitude of divergence
- confidence in baseline trajectory
- professional wording suitable for an economic report

Do NOT list numbers vertically (do NOT output one character per line).
Write in normal paragraph form.
Return one short paragraph.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"(LLM summary unavailable: {e})"

def generate_all_gdp_reports(countries=None):
    """
    Generate GDP reports for a list of countries.
    Returns list of output file paths.
    """
    if countries is None:
        countries = DEFAULT_GDP_COUNTRIES

    paths = []
    for ctry in countries:
        print(f"Generating GDP report for {ctry} ...")
        path = generate_gdp_report(ctry)
        paths.append(path)
    return paths