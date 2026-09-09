import os
import logging
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize module-level logger
logger = logging.getLogger(__name__)

class ServiceNowClient:

    def __init__(self):
        self.base_url = os.getenv("SN_BASE_URL")
        self.username = os.getenv("SN_USERNAME")
        self.password = os.getenv("SN_PASSWORD")
        
        if not all([self.base_url, self.username, self.password]):
            logger.critical("ServiceNow credentials (SN_BASE_URL, SN_USERNAME, SN_PASSWORD) missing in .env")
            raise ValueError("ServiceNow credentials must be set in .env")

        self.session = requests.Session()
        self.session.auth = (self.username, self.password) # type: ignore
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        logger.debug("ServiceNowClient initialized successfully.")

    def _fetch_incidents(self, query: str, fields: List[str], display_value: bool, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Generic private method to handle ServiceNow REST calls with dynamic queries and fields.
        """
        url = f"{self.base_url}/api/now/table/incident"
        
        params = {
            "sysparm_query": query,
            "sysparm_fields": ",".join(fields),
            "sysparm_display_value": "true" if display_value else "false",
            "sysparm_exclude_reference_link": "true",
            "sysparm_limit": limit
        }
        
        try:
            logger.info(f"Fetching incidents from ServiceNow with query: {query}")
            logger.debug(f"Request params: {params}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            results = response.json().get("result", [])
            logger.info(f"Successfully retrieved {len(results)} incidents.")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ServiceNow API Error (Query: {query}): {e}", exc_info=True)
            return []

    def get_incidents_for_indexing(self, updated_since: str, custom_fields: Optional[List[str]] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        1. Snapshot Retrieval: Fetches lightweight incident data created/updated after a certain date.
        Used to generate the semantic embeddings for the vector store.
        """

        fields = ["sys_id", "number", "short_description", "description", "cmdb_ci", "category", "subcategory"]        

        if custom_fields:
            fields = list(set(fields + custom_fields))
            
        query = f"sys_updated_on>={updated_since}"
        
        return self._fetch_incidents(query=query, fields=fields, display_value=True, limit=limit)

    def get_incidents_for_investigation(self, incident_numbers: List[str], custom_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:

        """
        2. Agent Investigation Tool: Fetches deep, rich context for specific incidents.
        Used by the LLM to read the historical investigation evidence.
        Fetches each incident individually to ensure complete extraction of journal fields.
        """

        # Detailed fields containing vital investigation history, metadata, and citations
        fields = [
            "number", 
            "short_description", 
            "description", 
            "work_notes",           
            "comments",             
            "close_notes",          
            "resolution_code",      
            "assignment_group", 
            "assigned_to",
            "resolved_at",
            "cmdb_ci",              
            "state",
            "impact",
            "urgency",
            "priority",
            "category",
            "subcategory",
            "caller_id",            
            "resolved_by"
        ]
        
        if custom_fields:
            fields = list(set(fields + custom_fields))
            
        detailed_incidents = []
        
        for number in incident_numbers:
            logger.info(f"Fetching deep investigation context for incident: {number}")
            query = f"number={number}"
            
            result = self._fetch_incidents(query=query, fields=fields, display_value=True, limit=1)
            
            if result:
                detailed_incidents.extend(result)
            else:
                logger.warning(f"Incident {number} not found or no data returned during deep fetch.")
                
        return detailed_incidents