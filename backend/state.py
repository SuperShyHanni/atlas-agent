from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from operator import add


def dict_reducer(a: Dict, b: Dict) -> Dict:
    if not a:
        return b.copy() if b else {}
    if not b:
        return a.copy()
    merged = a.copy()
    for key, value in b.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged


class AcademicState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    profile: Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks: Annotated[Dict, dict_reducer]
    results: Annotated[Dict[str, Any], dict_reducer]
    current_agent: str
    session_id: str
