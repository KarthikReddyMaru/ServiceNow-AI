import os
import httpx
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv()

class LLMProvider:
    """
    Isolated factory class to provide Chat models.
    Supports OpenAI, Ollama, or any OpenAI-compatible API.
    """
    @staticmethod
    def get_openai_chat(temperature: float = 0.0) -> ChatOpenAI:
        model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        # Configure custom sync and async HTTPX clients with TLS disabled for local dev
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            http_client=http_client,
            http_async_client=http_async_client
        )