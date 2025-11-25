import streamlit as st
import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.query_engine import answer_query
from src.report.pdf_report import DEFAULT_GDP_COUNTRIES

st.title("💬 Ask the Report (Agentic AI)")

country = st.selectbox("Select Country", DEFAULT_GDP_COUNTRIES)

question = st.text_area("Ask a question about the report:", height=120)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = answer_query(country, question)
        st.success("Answer:")
        st.write(response)
