import os
import shutil
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_chroma import Chroma
from .provider import EmbeddingProvider
from .formatter import IncidentFormatter

load_dotenv()
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = "knowledge_base"

class IncidentVectorStore:
    """
    Manages the ChromaDB instance and the local file-based knowledge base for incidents.
    """
    def __init__(self):
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "servicenow_incidents")
        self.embeddings = EmbeddingProvider.get_openai_embeddings()
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.debug(f"Initialized ChromaDB in {self.persist_directory} [Collection: {self.collection_name}]")

    @staticmethod
    def clear_workspace():
        """
        Aggressively deletes local storage folders to provide a clean slate.
        Must be called BEFORE initializing the class to avoid Windows file locks on Chroma's SQLite DB.
        """
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        for directory in [KNOWLEDGE_BASE_DIR, chroma_dir]:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    logger.info(f"Deleted directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to delete {directory}: {e}")

    def process_and_store_incidents(self, incidents: List[Dict[str, Any]]):
        """
        Takes raw ServiceNow incidents, delegates formatting, saves them to disk, 
        and embeds them in the vector database in a single batch.
        """
        if not incidents:
            logger.info("No incidents provided to store.")
            return

        documents_to_embed = []
        document_ids = []
        success_count = 0

        for incident in incidents:
            number = incident.get("number")
            if not number:
                continue
                
            # Delegate all formatting to the Formatter module
            doc = IncidentFormatter.to_document(incident)
            
            # Save Markdown file to disk
            incident_dir = os.path.join(KNOWLEDGE_BASE_DIR, number)
            os.makedirs(incident_dir, exist_ok=True)
            file_path = os.path.join(incident_dir, f"{number}.md")
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(doc.page_content)
                
                documents_to_embed.append(doc)
                document_ids.append(number)
                
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process incident {number}: {e}")

        logger.info(f"Saved {success_count}/{len(incidents)} incident markdown files to '{KNOWLEDGE_BASE_DIR}/'.")

        # Perform the batch upsert to ChromaDB
        if documents_to_embed:
            logger.info(f"Generating embeddings and indexing {len(documents_to_embed)} chunks in ChromaDB...")
            self.vectorstore.add_documents(documents=documents_to_embed, ids=document_ids)
            logger.info("Vector store ingestion complete.")

    def get_retriever(self, top_k: int = 5):
        """Returns a LangChain Retriever instance."""
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})