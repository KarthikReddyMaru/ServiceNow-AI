import logging
from typing import List, Optional, Any

import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from servicenow_ai.llm.assistant import InvestigationAgent

logger = logging.getLogger(__name__)


def _convert_history_to_langchain(history: List[Any]) -> List[BaseMessage]:
    """
    Converts Gradio conversation history to LangChain BaseMessage objects.
    Resilient to modern Gradio dict format [{"role": "...", "content": "..."}]
    and legacy tuple format [(user_msg, ai_msg)].
    """
    messages: List[BaseMessage] = []
    if not history:
        return messages

    for entry in history:
        # Modern Gradio format (list of dicts)
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                messages.append(AIMessage(content=str(content)))
        # Legacy tuple format (user, bot)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_text, bot_text = entry
            if user_text:
                messages.append(HumanMessage(content=str(user_text)))
            if bot_text:
                messages.append(AIMessage(content=str(bot_text)))

    return messages


def create_ui(agent: Optional[InvestigationAgent] = None) -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks UI instance.
    Decoupled from execution/launching for scalability and testing.
    """
    if agent is None:
        agent = InvestigationAgent()

    def respond(message: str, history: List[Any]):
        """Callback invoked by ChatInterface on every user submission."""
        logger.info(f"UI received prompt: '{message}'")
        langchain_history = _convert_history_to_langchain(history)
        return agent.chat(user_input=message, chat_history=langchain_history)

    with gr.Blocks(title="ServiceNow AI Incident Investigator", fill_height=True) as demo:
        gr.Markdown(
            """
            # 🛠️ ServiceNow Incident Investigation Copilot
            *Autonomous root cause analysis backed by historical ServiceNow incidents.*
            """
        )

        gr.ChatInterface(
            fn=respond,
            textbox=gr.Textbox(
                placeholder="Describe current symptoms (e.g. '504 gateway timeout') or enter an incident number (e.g. 'INC0010001')...",
                scale=8,
                container=False,
            ),
            examples=[
                "Payment API is returning HTTP 504 gateway timeout after today's release",
                "Can you check the details and resolution for INC0010001?",
                "Database connection pool exhaustion on production checkout pods",
                "High latency and intermittent connection drops reported on MySQL database",
            ],
            cache_examples=False,
        )

    return demo