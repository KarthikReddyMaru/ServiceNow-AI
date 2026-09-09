import os
import httpx
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Suppress insecure request warnings caused by verify=False
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

load_dotenv()

class EmbeddingProvider:

    """
    Isolated factory class to provide embedding models. 
    Can be swapped easily without affecting the vector store logic.
    """

    @staticmethod
    def get_openai_embeddings() -> OpenAIEmbeddings:

        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        
        # Configure custom sync and async HTTPX clients with TLS disabled
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        
        return OpenAIEmbeddings(
            base_url = base_url,
            model = model_name,
            http_client = http_client,
            http_async_client = http_async_client,
            check_embedding_ctx_length = False
        )