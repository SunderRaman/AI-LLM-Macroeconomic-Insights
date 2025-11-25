"""
economist_llm.py (Pipeline)
---------------------------
Dynamic economic answer generator with style adaptation and
automatic evaluation scoring. Designed to work with Gemini 2.5 Flash
via the `GeminiLLMClient` in src/llm/models.

Provides:
- generate_answer(context, question)
- evaluate_answer(context, question, answer)
- ask(context, question)  # wrapper
"""

from dataclasses import dataclass
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LLMResult:
    answer: str
    score: Dict[str, float]


class EconomistLLM:
    def __init__(self, llm_client):
        """
        llm_client: must provide `.complete(prompt: str) -> str`
        Compatible with GeminiLLMClient.
        """
        self.llm_client = llm_client

    def _build_dynamic_prompt(self, context: str, question: str) -> str:
        return f"""
You are an economist AI assistant.

Use ONLY the following context:

{context}

RULES:
- GDP values come ONLY from baseline forecast
- MAPE reflects uncertainty
- Scenario values represent deviations, not actual GDP
- PDF text may include expert interpretation

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
- 3–4 sentences
- summarize baseline trend
- integrate uncertainty reasoning
- include scenario insight if relevant (e.g., downside risk or upside potential)
- professional, concise tone

TYPE 3 OUTPUT:
- 4–5 sentences
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

    def _clean_answer(self, text: str) -> str:
        """Removes markdown, excess whitespace, artifacts."""
        text = re.sub(r"```[a-zA-Z]*", "", text)  # remove code fences
        text = text.replace("**", "")            # remove bold markers
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate_answer(self, context: str, question: str) -> str:
        prompt = self._build_dynamic_prompt(context, question)
        raw = self.llm_client.complete(prompt)
        answer = self._clean_answer(raw)
        logger.info(f"Generated answer: {answer}")
        return answer

    def _build_evaluation_prompt(self, context: str, question: str, answer: str) -> str:
        return f"""
Evaluate the following answer based solely on the provided context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Score the answer on a scale of 1–5 for each dimension:
1. Accuracy (correct use of data from context)
2. Structure (matches required style: 2 / 3–4 / 4–5 sentences)
3. Insight (adds meaningful interpretation)
4. Clarity and professionalism

Provide output strictly in JSON format:
{{
  "accuracy": <number>,
  "structure": <number>,
  "insight": <number>,
  "clarity": <number>
}}
"""

    def evaluate_answer(self, context: str, question: str, answer: str) -> Dict[str, float]:
        eval_prompt = self._build_evaluation_prompt(context, question, answer)
        raw_eval = self.llm_client.complete(eval_prompt)
        clean = raw_eval.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()
        
        try:
            import json
            scores = json.loads(clean)
        except Exception as e:
            logger.error(f"Evaluation parsing failed: {e}\nRaw eval: {raw_eval}")
            scores = {
                "accuracy": 3,
                "structure": 3,
                "insight": 3,
                "clarity": 3,
            }

        overall = (
            scores.get("accuracy", 0) * 0.4
            + scores.get("structure", 0) * 0.3
            + scores.get("insight", 0) * 0.2
            + scores.get("clarity", 0) * 0.1
        )
        scores["overall"] = round(overall, 2)

        logger.info(f"Evaluation scores: {scores}")
        return scores

    def ask(self, context: str, question: str) -> LLMResult:
        answer = self.generate_answer(context, question)
        score = self.evaluate_answer(context, question, answer)

        logger.info(f"Final Answer: {answer}")
        logger.info(f"Final Score: {score}")

        return LLMResult(answer=answer, score=score)


# Example usage placeholder
if __name__ == "__main__":
    print("EconomistLLM ready — integrate with GeminiLLMClient from src/llm/models.")
