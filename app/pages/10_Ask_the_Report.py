import streamlit as st
import sys, os
import google.generativeai as genai
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.agent.retriever import retrieve_context # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


st.title("Ask the Report (Agentic AI)")

country = st.selectbox(
    "Select Country",
    ["India", "United States", "China", "United Kingdom", "Germany", "Japan"]
)

question = st.text_input("Ask a question about the report")

if st.button("Ask"):
    context = retrieve_context(question, country)

    prompt = f"""
        You are an economist AI assistant.

        Use ONLY the following context:

        {context}

        RULES:
        - GDP values come ONLY from baseline forecast
        - MAPE reflects uncertainty
        - Scenario values represent deviations, not actual GDP
        - PDF text may include expert interpretation
       
        Formatting requirements:
        - One paragraph only
        - Plain text output
        - No Markdown
        - No italics or bold
        - Do NOT break words across lines
        - Keep numbers and units together (e.g., "6.8 trillion")
        - No character-by-character output

        STYLE LOGIC:
        First, classify the question into one of the following types:
        1. Quantitative/ranking (asks for highest, lowest, largest, MAPE values, comparisons)
        2. Interpretive/trend (asks about trends, drivers, outlooks, implications)
        3. Scenario/policy impact (asks about downside/upside effects, shocks, policy changes)

        Then apply the corresponding style:

        TYPE 1 OUTPUT:
        - 2 sentences
        - include key metric value (MAPE % or GDP change)
        - brief comparison to at least one peer
        - single implication for forecast reliability

        TYPE 2 OUTPUT:
        - 3-4 sentences
        - summarize baseline trend
        - integrate uncertainty reasoning
        - include scenario insight if relevant (e.g., downside risk or upside potential)
        - professional, concise tone

        TYPE 3 OUTPUT:
        - 4-5 sentences
        - contrast baseline and scenario outcomes
        - explain mechanism of impact
        - highlight uncertainty amplification
        - professional report style

        GENERAL RULES:
        - no bullet points
        - no restating the question
        - avoid repetition of terms
        - maximum 110 words

        TASK:
        Answer the question using the above logic.

        Question: {question}
    """


    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    st.write(response.text)
