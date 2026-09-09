import logging
import gradio as gr
from gradio import themes  # Explicitly import themes to satisfy Pylance
from dotenv import load_dotenv

from servicenow_ai.ui.chat import create_ui

# Configure logging for the root application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()

def main():
    demo = create_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=themes.Soft(  # Use the explicit import here
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )

if __name__ == "__main__":
    main()