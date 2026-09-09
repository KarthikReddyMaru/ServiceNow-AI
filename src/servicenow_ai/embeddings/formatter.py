from typing import Dict, Any
from langchain_core.documents import Document

class IncidentFormatter:
    """
    Transforms raw ServiceNow incident dictionaries into semantic representations 
    (Markdown strings and LangChain Documents) optimized for vector embeddings.
    """
    
    @staticmethod
    def to_markdown(incident: Dict[str, Any]) -> str:
        """Formats raw ServiceNow JSON into a dense, Markdown-formatted semantic string."""
        number = incident.get("number", "Unknown")
        sys_id = incident.get("sys_id", "Unknown")
        category = incident.get("category", "None specified")
        subcategory = incident.get("subcategory", "None specified")
        ci = incident.get("cmdb_ci", "None specified")
        short_desc = incident.get("short_description", "").strip()
        desc = incident.get("description", "").strip()
        
        # Truncate massive stack traces for vector compatibility
        if len(desc) > 1000:
            desc = desc[:1000] + "\n... [Truncated for index. Full logs retrieved at runtime]"

        chunk = f"# Incident: {number}\n\n"
        chunk += f"__SysID:__ {sys_id}\n\n"
        chunk += f"__Category:__ {category}\n\n"
        chunk += f"__Subcategory:__ {subcategory}\n\n"
        chunk += f"__Configuration Item:__ {ci}\n\n"
        chunk += f"__Summary:__ {short_desc}\n\n"
        chunk += "__Description:__\n\n"
        chunk += f"{desc}"
        
        return chunk.strip()

    @classmethod
    def to_document(cls, incident: Dict[str, Any]) -> Document:
        """Constructs a LangChain Document with the markdown content and strictly typed metadata."""
        markdown_content = cls.to_markdown(incident)
        
        return Document(
            page_content=markdown_content,
            metadata={
                "incident_number": incident.get("number", "Unknown"),
                "sys_id": str(incident.get("sys_id", "")),
                "category": str(incident.get("category", "")),
                "subcategory": str(incident.get("subcategory", "")),
                "cmdb_ci": str(incident.get("cmdb_ci", ""))
            }
        )