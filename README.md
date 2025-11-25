# Macro AI Project – Economist LLM with Gemini

This repository contains a macroeconomic analysis assistant built on top of a Large Language Model (LLM) and Gemini 2.5 Flash.  
It generates professional, IMF/OECD-style insights from baseline GDP forecasts, backtesting metrics (MAPE), and scenario data.

The system:
- Adapts its answer style based on the type of question
- Uses only the provided numerical and text context
- Avoids hallucinations and explicitly states when data is missing
- Automatically scores the quality of each answer

---

## Features

- **Dynamic answer styles**
  - Type 1: Quantitative/ranking questions → short 2-sentence answer with metrics + comparison
  - Type 2: Interpretive/trend questions → 3–4 sentence outlook with uncertainty reasoning
  - Type 3: Scenario/policy impact questions → 4–5 sentence baseline vs scenario analysis

- **Evaluation engine**
  - Secondary LLM pass to score answers on:
    - Accuracy
    - Structure
    - Insight
    - Clarity
  - Computes a weighted overall score (0–5)

- **Safe numeric reasoning**
  - Uses baseline forecasts from CSV
  - Uses MAPE as a proxy for forecast uncertainty
  - Uses scenario data (downside/upside) only when present
  - Refuses to fabricate scenario impacts when data is missing

- **Gemini 2.5 Flash integration**
  - Uses `google-generativeai`
  - API key loaded from `.env`
  - Simple `complete(prompt: str) -> str` interface

---

## Project Structure

```text
macro_ai_project/
├─ src/
│  ├─ __init__.py
│  ├─ llm/
│  │  ├─ __init__.py
│  │  ├─ models/
│  │  │  ├─ __init__.py
│  │  │  └─ gemini_client.py         # Gemini 2.5 Flash wrapper
│  │  └─ pipelines/
│  │     ├─ __init__.py
│  │     └─ economist_llm.py         # Dynamic economist LLM + evaluation
│  └─ test_macro_ai.py               # Example script to test the pipeline
├─ .env.example                      # Template for environment variables
├─ .gitignore
├─ README.md
└─ requirements.txt                  # Python dependencies (to be filled)
