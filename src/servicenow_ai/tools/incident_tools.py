import json
import logging
from typing import List
from langchain_core.tools import tool

from servicenow_ai.embeddings.retrieval import IncidentRetrievalService
from src.servicenow_ai.embeddings.vector_store import IncidentVectorStore
from src.servicenow_ai.client.servicenow import ServiceNowClient

logger = logging.getLogger(__name__)

# Initialize the underlying clients once when the module loads
vector_store = IncidentVectorStore()
sn_client = ServiceNowClient()

# Wrap the vector store in our retrieval service which handles query optimization
retrieval_service = IncidentRetrievalService(vector_store)

@tool
def search_historical_incident_ids(user_query: str, limit: int = 5) -> str:
    """
    Search historical incidents based on a natural language description of the current issue.
    
    Args:
        user_query: A description of the problem, symptoms, or error messages. 
                    (The tool will automatically optimize this query for vector search).
        limit: Maximum number of incident IDs to return (default is 5).
        
    Returns:
        A JSON string containing the relevant incident numbers and high-level metadata (Service, CI). 
        You MUST use the fetch_incident_investigation_details tool to read the actual resolution and work notes of these incidents.
    """
    logger.info(f"Tool executed: search_historical_incident_ids for raw query: '{user_query}'")
    
    # We pass an empty chat_history here because the main Agent maintains the conversational context.
    # The internal retrieval service just needs to optimize the specific search intent.
    docs, optimized_query = retrieval_service.retrieve(
        query=user_query, 
        chat_history=[], 
        fetch_k=10, 
        return_k=limit
    )
    
    logger.info(f"Internal Vector Search executed using optimized query: '{optimized_query}'")
    
    results = []
    for doc in docs:
        inc_number = doc.metadata.get("incident_number")
        if inc_number:
            results.append({
                "incident_number": inc_number,
                "category": doc.metadata.get("category", "Unknown"),
                "subcategory": doc.metadata.get("subcategory", "Unknown"),
                "configuration_item": doc.metadata.get("cmdb_ci", "Unknown"),
                "brief_preview": doc.page_content[:150].replace("\n", " ") + "..."
            })
            
    if not results:
        return json.dumps({"message": "No relevant historical incidents found."})
        
    return json.dumps(results, indent=2)

@tool
def fetch_incident_investigation_details(incident_numbers: List[str]) -> str:
    """
    Fetch the complete, authoritative investigation evidence directly from ServiceNow.
    
    Args:
        incident_numbers: A list of specific incident strings (e.g., ["INC0010001", "INC0010002"]).

    Returns:
        A formatted text payload containing the real engineer work notes, comments, and resolution codes from ServiceNow.
    """
    logger.info(f"Tool executed: fetch_incident_investigation_details for IDs: {incident_numbers}")
    
    if not incident_numbers:
        return "Error: No incident numbers provided."

    incidents = sn_client.get_incidents_for_investigation(incident_numbers=incident_numbers)
    
    if not incidents:
        return f"Could not fetch data for incidents: {incident_numbers}. They may not exist or access is denied."

    evidence_dossier = []
    
    for inc in incidents:
        # Extract metadata
        number = inc.get("number", "Unknown")
        state = inc.get("state", "Unknown")
        priority = inc.get("priority", "Unknown")
        impact = inc.get("impact", "Unknown")
        urgency = inc.get("urgency", "Unknown")
        
        category = inc.get("category", "Unknown")
        subcategory = inc.get("subcategory", "Unknown")
        ci = inc.get("cmdb_ci", "Unknown")
        caller = inc.get("caller_id", "Unknown")
        
        # Extract Assignment & Resolution
        assignment_group = inc.get("assignment_group", "Unassigned")
        assigned_to = inc.get("assigned_to", "Unassigned")
        resolved_by = inc.get("resolved_by", "Unknown")
        resolved_at = inc.get("resolved_at", "Unknown")
        resolution_code = inc.get("resolution_code", "None")
        
        # Extract Core Text
        short_desc = inc.get("short_description", "None")
        desc = inc.get("description", "None")
        close_notes = inc.get("close_notes", "No close notes recorded.")
        work_notes = inc.get("work_notes", "No work notes recorded.")
        comments = inc.get("comments", "No external comments recorded.")
        
        dossier_entry = f"""
            ==================================================
            INCIDENT EVIDENCE: {number} 
            ==================================================
            [CORE METADATA]
            State: {state}
            Priority: {priority} (Impact: {impact}, Urgency: {urgency})
            Category: {category} / Subcategory: {subcategory}
            Configuration Item: {ci}
            Caller: {caller}

            [ASSIGNMENT & RESOLUTION]
            Assignment Group: {assignment_group}
            Assigned To: {assigned_to}
            Resolved By: {resolved_by} (At: {resolved_at})
            Resolution Code: {resolution_code}

            [PROBLEM DESCRIPTION]
            Summary: {short_desc}
            Description: 
            {desc}

            [CLOSE NOTES / FINAL RESOLUTION]
            {close_notes}

            [ENGINEER WORK NOTES (Internal)]
            {work_notes}

            [ADDITIONAL COMMENTS (Customer Facing)]
            {comments}
            ==================================================
        """
        evidence_dossier.append(dossier_entry.strip())

    return "\n\n".join(evidence_dossier)