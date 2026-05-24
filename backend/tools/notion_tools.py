"""
Notion integration tools — read and write student notes/pages via Notion API.
Requires NOTION_API_KEY in environment.
"""
import os
from langchain_core.tools import tool

_client = None


def _get_client():
    global _client
    if _client is None:
        from notion_client import Client
        api_key = os.getenv("NOTION_API_KEY", "")
        if not api_key:
            raise RuntimeError("NOTION_API_KEY not set in environment")
        _client = Client(auth=api_key)
    return _client


@tool
def notion_search(query: str) -> str:
    """Search pages and databases in the student's Notion workspace by keyword."""
    try:
        client = _get_client()
        results = client.search(query=query, page_size=5).get("results", [])
        if not results:
            return "No Notion pages found matching that query."
        lines = []
        for r in results:
            title = _extract_title(r)
            url = r.get("url", "")
            obj_type = r.get("object", "")
            lines.append(f"[{obj_type}] {title} — {url}")
        return "\n".join(lines)
    except Exception as e:
        return f"Notion search error: {e}"


@tool
def notion_get_page(page_id: str) -> str:
    """Retrieve the text content of a Notion page by its ID or URL."""
    try:
        client = _get_client()
        # Accept full URLs — extract the ID from the last path segment
        if page_id.startswith("http"):
            page_id = page_id.rstrip("/").split("-")[-1].split("/")[-1]
        blocks = client.blocks.children.list(block_id=page_id).get("results", [])
        lines = []
        for block in blocks:
            text = _extract_block_text(block)
            if text:
                lines.append(text)
        return "\n".join(lines) if lines else "(Page is empty or has no text blocks)"
    except Exception as e:
        return f"Notion get page error: {e}"


@tool
def notion_create_page(parent_page_id: str, title: str, content: str) -> str:
    """
    Create a new Notion page under the given parent page.
    Use this to save study notes, summaries, or study plans.
    parent_page_id: the ID or URL of the parent page.
    """
    try:
        client = _get_client()
        if parent_page_id.startswith("http"):
            parent_page_id = parent_page_id.rstrip("/").split("-")[-1].split("/")[-1]

        paragraphs = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": line}}]
                },
            }
            for line in content.split("\n")
            if line.strip()
        ]

        page = client.pages.create(
            parent={"type": "page_id", "page_id": parent_page_id},
            properties={
                "title": {"title": [{"type": "text", "text": {"content": title}}]}
            },
            children=paragraphs[:100],  # Notion API limit per request
        )
        return f"Page created: {page.get('url', 'unknown URL')}"
    except Exception as e:
        return f"Notion create page error: {e}"


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_title(obj: dict) -> str:
    props = obj.get("properties", {})
    for key in ("title", "Name", "Title"):
        if key in props:
            rich = props[key].get("title", [])
            return "".join(t.get("plain_text", "") for t in rich)
    return "(untitled)"


def _extract_block_text(block: dict) -> str:
    btype = block.get("type", "")
    content = block.get(btype, {})
    rich = content.get("rich_text", [])
    return "".join(t.get("plain_text", "") for t in rich)
