import os
import sys
import pandas as pd
import streamlit as st
import google.generativeai as genai
import zipfile
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.report.pdf_report import generate_gdp_report, generate_all_gdp_reports, DEFAULT_GDP_COUNTRIES


# Load data
FC_PATH = "data/processed/forecasts/forecasts_all.csv"
BT_PATH = "data/processed/backtesting/backtest_metrics_by_horizon.csv"


@st.cache_data
def load_data():
    fc = pd.read_csv(FC_PATH)
    bt = pd.read_csv(BT_PATH)
    return fc, bt

def generate_summary_gemini(api_key, country, indicator, fc_row, bt_rows):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are an expert macroeconomic analyst writing a concise professional insight.

Country: {country}
Indicator: {indicator}

Forecast:
{fc_row.to_string(index=False)}

Backtesting accuracy (MAPE by horizon %):
{bt_rows.to_string(index=False)}

Write a clear 4-6 sentence summary including:

Formatting requirements:
- One paragraph only
- Plain text output
- No Markdown
- No italics or bold
- Do NOT break words across lines
- Keep numbers and units together (e.g., "6.8 trillion")
- No character-by-character output


Avoid technical jargon. Tone: professional, clear, concise.
"""

    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Module E — AI Insights (Gemini)")

    api_key = st.text_input("Enter Gemini API Key", type="password")
    if not api_key:
        st.warning("Please enter your Gemini API key")
        return

    fc, bt = load_data()

    countries = fc["country"].unique().tolist()
    indicators = fc["indicator"].unique().tolist()

    country = st.selectbox("Country", countries, index=countries.index("India"))
    indicator = st.selectbox("Indicator", indicators, index=indicators.index("gdp_current_usd"))

    fc_sel = fc[(fc["country"] == country) & (fc["indicator"] == indicator)]
    fc_latest = fc_sel.sort_values("year").tail(3)

    bt_sel = bt[(bt["country"] == country) & (bt["indicator"] == indicator)]

    if st.button("Generate Insight"):
        with st.spinner("Generating macro insight..."):
            text = generate_summary_gemini(
                api_key,
                country,
                indicator,
                fc_latest,
                bt_sel[["horizon","mape_pct"]]
            )
        st.subheader("AI Insight")
        st.write(text)

    country = st.selectbox("Country", ["India", "China", "United States", "Germany", "Japan", "United Kingdom"], index=0)

    if st.button("Generate PDF Report"):
        path = generate_gdp_report(country)
        with open(path, "rb") as f:
            st.download_button(
                label="Download GDP Report PDF",
                data=f,
                file_name=path.split(os.sep)[-1],
                mime="application/pdf",
            )    

    st.markdown("---")
    st.header("📄 Export Reports")

    # Single-country report
    st.subheader("Single-country GDP Report")

    country_for_pdf = st.selectbox(
        "Select country for PDF export",
        DEFAULT_GDP_COUNTRIES,
        index=DEFAULT_GDP_COUNTRIES.index("India") if "India" in DEFAULT_GDP_COUNTRIES else 0,
    )

    if st.button("Generate PDF for selected country"):
        path = generate_gdp_report(country_for_pdf)
        st.success(f"Generated report: {os.path.basename(path)}")

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=os.path.basename(path),
            mime="application/pdf",
        )

    # Multi-country batch export
    st.subheader("Multi-country GDP Reports (ZIP)")

    if st.button("Generate all GDP reports (batch)"):
        paths = generate_all_gdp_reports()
        st.success(f"Generated {len(paths)} reports.")

        # Build ZIP in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                arcname = os.path.basename(p)
                zf.write(p, arcname=arcname)
        zip_buf.seek(0)

        st.download_button(
            label="Download all reports (ZIP)",
            data=zip_buf.getvalue(),
            file_name="macro_gdp_reports.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    main()
