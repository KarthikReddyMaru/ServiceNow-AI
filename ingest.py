import os
import argparse
import logging
from datetime import datetime
import sys

# Add the project root to the Python path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.servicenow_ai.client.servicenow import ServiceNowClient

# Setup logging for the script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = "knowledge_base"

def format_incident_to_chunk(incident: dict) -> str:
    """
    Sanitizes and formats the raw ServiceNow JSON into a dense, Markdown-formatted 
    semantic representation. This ensures optimal embedding quality.
    """
    number = incident.get("number", "Unknown")
    sys_id = incident.get("sys_id", "Unknown")
    service = incident.get("business_service", "None specified")
    ci = incident.get("cmdb_ci", "None specified")
    short_desc = incident.get("short_description", "").strip()
    desc = incident.get("description", "").strip()
    
    if len(desc) > 1000:
        desc = desc[:1000] + "\n... [Truncated for index. Full logs retrieved at runtime]"

    chunk = f"""
# Incident: {number}

 __SysID:__ {sys_id}

__Service:__ {service}

__Configuration Item:__ {ci}

__Summary:__ {short_desc}

__Description:__

{desc}
"""

    return chunk.strip()

def main():
    parser = argparse.ArgumentParser(description="Ingest ServiceNow incidents for the Vector DB.")
    parser.add_argument("--start-date", type=str, help="Date to fetch from (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--clear", action="store_true", help="Clear the vector database before ingestion.")
    args = parser.parse_args()

    if args.clear:
        logger.info("Flag --clear passed. [Placeholder]: Vector DB will be cleared here.")
        # TODO: Implement vector DB clearing logic
        pass

    start_date = args.start_date
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    
    sn_query_date = f"{start_date} 00:00:00"
    logger.info(f"Starting ingestion for incidents updated >= {sn_query_date}")

    client = ServiceNowClient()
    incidents = client.get_incidents_for_indexing(updated_since=sn_query_date)

    if not incidents:
        logger.info("No incidents found for the specified date range.")
        return

    logger.info(f"Fetched {len(incidents)} incidents. Formatting and saving to disk...")

    success_count = 0
    for incident in incidents:
        number = incident.get("number")
        if not number:
            continue
            
        formatted_chunk = format_incident_to_chunk(incident)
        
        # Create a dedicated directory for the incident
        incident_dir = os.path.join(KNOWLEDGE_BASE_DIR, number)
        os.makedirs(incident_dir, exist_ok=True)
        
        # Save as Markdown file. "w" mode overwrites it if it already exists.
        file_path = os.path.join(incident_dir, f"{number}.md")
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_chunk)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to write file for {number}: {e}")

    logger.info(f"Ingestion complete. Saved {success_count}/{len(incidents)} incident chunks to '{KNOWLEDGE_BASE_DIR}/'.")

if __name__ == "__main__":
    main()