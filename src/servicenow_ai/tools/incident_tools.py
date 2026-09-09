import json
import logging
from typing import List
from langchain_core.tools import tool

from servicenow_ai.embeddings.vector_store import IncidentVectorStore
from servicenow_ai.client.servicenow import ServiceNowClient
from servicenow_ai.embeddings.retrieval import IncidentRetrievalService

logger = logging.getLogger(__name__)

vector_store = IncidentVectorStore()
sn_client = ServiceNowClient()
retrieval_service = IncidentRetrievalService(vector_store)


def _fetch_and_format_dossier(incident_numbers: List[str]) -> str:
    """
    Core business logic to fetch incidents from ServiceNow and format them 
    into a markdown dossier. Not exposed directly to the LLM.
    """
    if not incident_numbers:
        return "Error: No incident numbers provided."

    incidents = sn_client.get_incidents_for_investigation(incident_numbers=incident_numbers)
    
    if not incidents:
        return f"Could not fetch data for incidents: {incident_numbers}. They may not exist or access is denied."

    evidence_dossier = []
    
    for inc in incidents:
        number = inc.get("number", "Unknown")
        state = inc.get("state", "Unknown")
        priority = inc.get("priority", "Unknown")
        impact = inc.get("impact", "Unknown")
        urgency = inc.get("urgency", "Unknown")
        
        category = inc.get("category", "Unknown")
        subcategory = inc.get("subcategory", "Unknown")
        ci = inc.get("cmdb_ci", "Unknown")
        caller = inc.get("caller_id", "Unknown")
        
        assignment_group = inc.get("assignment_group", "Unassigned")
        assigned_to = inc.get("assigned_to", "Unassigned")
        resolved_by = inc.get("resolved_by", "Unknown")
        resolved_at = inc.get("resolved_at", "Unknown")
        resolution_code = inc.get("resolution_code", "None")
        
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


@tool
def search_historical_incident_ids(user_query: str, limit: int = 5) -> str:
    """
    Search historical incidents based on a natural language description of the current issue.
    
    Args:
        user_query: A description of the problem, symptoms, or error messages.
        limit: Maximum number of incident IDs to return (default is 5).
        
    Returns:
        JSON string containing incident numbers. You MUST use fetch_incident_investigation_details 
        afterwards to read the actual resolution.
    """
    logger.info(f"Tool executed: search_historical_incident_ids for query: '{user_query}'")
    
    docs, optimized_query = retrieval_service.retrieve(
        query=user_query, 
        chat_history=[], 
        fetch_k=10, 
        return_k=limit
    )
    
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
    """
    logger.info(f"Tool executed: fetch_incident_investigation_details for IDs: {incident_numbers}")
    return _fetch_and_format_dossier(incident_numbers)


@tool
def lookup_incident_by_number(incident_number: str) -> str:
    """
    Use this tool when the user EXPLICITLY provides an incident number (e.g., "INC0010001").
    
    Args:
        incident_number: The exact incident number (e.g., "INC0010001").
    """
    logger.info(f"Tool executed: lookup_incident_by_number for {incident_number}")
    # Now we just call standard python, avoiding the invoke() type mismatch entirely
    return _fetch_and_format_dossier([incident_number])