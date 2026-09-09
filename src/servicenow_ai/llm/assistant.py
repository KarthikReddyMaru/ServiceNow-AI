import logging
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents import create_agent

from servicenow_ai.llm.provider import LLMProvider
from servicenow_ai.tools.incident_tools import (
    search_historical_incident_ids,
    fetch_incident_investigation_details,
    lookup_incident_by_number
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
    You are an elite AI Production Support Assistant integrated with ServiceNow.
    Your sole purpose is to help production support engineers investigate and resolve current system issues by analyzing historical incidents.

    ### STRICT SCOPE
    - You ONLY assist with production support, IT troubleshooting, and incident investigation. 
    - If the user asks about general knowledge, programming outside of a specific issue, or casual chat, politely decline and steer them back to incident resolution.

    ### YOUR CAPABILITIES
    1. If the user describes an issue/symptom: Use `search_historical_incident_ids` to find past occurrences, then ALWAYS use `fetch_incident_investigation_details` on those IDs to read how they were fixed.
    2. If the user asks about a specific incident ID: Use `lookup_incident_by_number` directly.

    ### OUTPUT REQUIREMENTS
    When providing a resolution based on historical data, you MUST:
    - **Cite the Incident:** Always mention the specific Incident ID you pulled the answer from.
    - **Provide Evidence:** Quote or summarize the specific troubleshooting steps from the engineer's work notes or close notes. Do not hallucinate fixes.
    - **Identify Personnel:** Explicitly state the `Assignment Group`, `Assigned To`, and `Resolved By` fields so the current engineer knows who to contact if they have doubts.
    - **Format:** Use Markdown for readability. Use bolding for key metrics, servers, or error codes.

    Example format for an insight:
    "Based on **INC0010045**, a similar issue occurred where... The issue was resolved by [fix]. 
    *Evidence:* According to the close notes, the engineer ran...
    *Contact:* This was resolved by John Doe (Database Admin Team). Reach out to them for verification."
"""

class InvestigationAgent:
    """
    The main agent that orchestrates the LLM and the ServiceNow tools.
    Designed to easily plug into Gradio's chat interface.
    """
    def __init__(self):
        self.llm = LLMProvider.get_openai_chat(temperature=0.2)
        
        self.tools = [
            search_historical_incident_ids,
            fetch_incident_investigation_details,
            lookup_incident_by_number
        ]
        
        # Using the latest LangChain harness for agent creation
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT
        )
        logger.info("InvestigationAgent initialized with create_agent.")

    def chat(self, user_input: str, chat_history: List[BaseMessage]) -> str:
        """
        The main entry point for the Gradio UI.
        
        Args:
            user_input: The latest string prompt from the user.
            chat_history: A list of LangChain Message objects representing the conversation so far.
            
        Returns:
            The final markdown string response from the AI.
        """
        logger.info(f"Agent received user input: {user_input}")
        
        messages = chat_history + [HumanMessage(content=user_input)]
        
        try:
            # The agent acts on the messages and manages the tool calling loop internally
            result = self.agent.invoke({"messages": messages}) # type: ignore
            
            # Extract the final answer text
            final_message = result["messages"][-1].content
            return final_message
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return "I encountered an internal error while trying to investigate this issue. Please check the backend logs."