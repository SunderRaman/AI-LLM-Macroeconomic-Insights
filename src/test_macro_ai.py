from src.llm.models.gemini_client import GeminiLLMClient
from src.llm.pipelines.economist_llm import EconomistLLM
from src.agent.retriever import retrieve_context 

# TEMP test values – replace with your real context & question
country = "India"
question = "How would a downside inflation scenario affect India’s long-term GDP outlook?"
context_list = retrieve_context(question, country)

# Convert list of context chunks into a single text block
context = "\n\n".join(context_list)


# question = "Which country shows the highest long-term forecast uncertainty?"

llm_client = GeminiLLMClient()
economist = EconomistLLM(llm_client)

result = economist.ask(context, question)

print("ANSWER:\n", result.answer)
print("\nSCORES:\n", result.score)