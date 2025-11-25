import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

class GeminiLLMClient:
    """
    Gemini 2.5 Flash LLM wrapper compatible with EconomistLLM.
    Loads API key from .env using python-dotenv.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ):
        # Load .env and pull API key
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please add it to your .env file:\n"
                "GEMINI_API_KEY=your_api_key_here"
            )

        genai.configure(api_key=api_key)

        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        # Instantiate model
        self.model = genai.GenerativeModel(self.model_name)

    def complete(self, prompt: str) -> str:
        """
        Standardized complete() interface expected by EconomistLLM.
        Returns clean text content from Gemini.
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )

            # Gemini returns .text when plain text
            if hasattr(response, "text") and response.text:
                return response.text.strip()

            # Fallback for safety
            return str(response).strip()

        except Exception as e:
            return f"Error generating response: {e}"