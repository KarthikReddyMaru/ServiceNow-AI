import os
import sys
import argparse
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.servicenow_ai.client.servicenow import ServiceNowClient
from src.servicenow_ai.embeddings.vector_store import IncidentVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest ServiceNow incidents for the Vector DB.")
    parser.add_argument("--start-date", type=str, help="Date to fetch from (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--clear", action="store_true", help="Clear the vector database and knowledge base before ingestion.")
    args = parser.parse_args()

    # Clear workspace BEFORE initializing VectorStore
    if args.clear:
        logger.warning("Flag --clear passed. Wiping local vector store and knowledge base directories...")
        IncidentVectorStore.clear_workspace()
        return

    start_date = args.start_date or datetime.now().strftime("%Y-%m-%d")
    sn_query_date = f"{start_date} 00:00:00"
    logger.info(f"Starting ingestion for incidents updated >= {sn_query_date}")

    client = ServiceNowClient()
    incidents = client.get_incidents_for_indexing(updated_since=sn_query_date)

    if not incidents:
        logger.info("No incidents found for the specified date range.")
        return

    logger.info(f"Fetched {len(incidents)} incidents from ServiceNow. Handing off to storage layer...")

    # Initialize Vector Store and process incidents
    vector_store = IncidentVectorStore()
    vector_store.process_and_store_incidents(incidents)

if __name__ == "__main__":
    main()