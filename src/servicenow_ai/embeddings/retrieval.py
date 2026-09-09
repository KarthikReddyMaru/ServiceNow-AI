import logging
from typing import List, Tuple
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from src.servicenow_ai.llm.provider import LLMProvider
from src.servicenow_ai.embeddings.vector_store import IncidentVectorStore

logger = logging.getLogger(__name__)

class SearchQuery(BaseModel):
    """Pydantic schema to enforce structured output from the LLM."""
    optimized_query: str = Field(
        description="The optimized search query for semantic vector search, extracting key technical symptoms, error codes, and system names."
    )

class IncidentRetrievalService:

    """
    Responsible for interpreting user intent, optimizing the search query, 
    and retrieving relevant historical incidents from the vector store.
    """

    def __init__(self, vector_store: IncidentVectorStore):
    
        self.llm = LLMProvider.get_openai_chat(temperature=0.0)
        self.vector_store = vector_store

        SYSTEM_PROMPT = """
            You are an expert IT incident investigator. 
            Given a conversation history and a new user query, formulate a single, standalone search query.
            This query will be used to search a vector database of historical ServiceNow incidents.
            Focus entirely on extracting technical keywords, error messages, component names, and symptoms.
            Do not include conversational filler.
        """
        
        self.query_rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        self.structured_llm = self.llm.with_structured_output(SearchQuery)
        self.rewrite_chain = self.query_rewrite_prompt | self.structured_llm

    def rewrite_query(self, query: str, chat_history: List[BaseMessage]) -> str:

        """Uses the LLM to rewrite the query contextually based on chat history."""

        try:
            logger.info("Rewriting user query for vector search optimization...")
            result = self.rewrite_chain.invoke({
                "chat_history": chat_history,
                "query": query
            })
            
            if isinstance(result, dict):
                optimized_query = result.get("optimized_query", query)
            else:
                optimized_query = getattr(result, "optimized_query", query)
                
            logger.debug(f"Optimized Query: {optimized_query}")
            return optimized_query
            
        except Exception as e:
            logger.error(f"Failed to rewrite query. Falling back to original query. Error: {e}", exc_info=True)
            return query

    def retrieve(self, query: str, chat_history: List[BaseMessage], fetch_k: int = 10, return_k: int = 5) -> Tuple[List[Document], str]:

        """
        Executes the retrieval pipeline:
        1. Rewrites the query.
        2. Fetches `fetch_k` candidates from the vector store.
        3. Reranks/Filters down to `return_k` results.
        
        Returns the final documents and the optimized query used.
        """
        optimized_query = self.rewrite_query(query, chat_history)
        
        # 1. Fetch Candidates (High Recall)
        logger.info(f"Fetching top {fetch_k} incidents from ChromaDB...")
        retriever = self.vector_store.get_retriever(top_k=fetch_k)
        docs = retriever.invoke(optimized_query)
        
        # 2. Rerank / Filter (High Precision)
        # TODO: Implement Cross-Encoder reranking here in the future (e.g., using bge-reranker).
        # For POC V1, we simply slice the top `return_k` results natively scored by ChromaDB.
        logger.info(f"Slicing down to top {return_k} best matches...")
        final_docs = docs[:return_k]
        
        return final_docs, optimized_query