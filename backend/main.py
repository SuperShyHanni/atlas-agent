import json
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from graph import graph
from state import AcademicState

app = FastAPI(title="ATLAS Academic Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: Dict[str, AcademicState] = {}


def get_or_create_session(session_id: str) -> AcademicState:
    if session_id not in sessions:
        sessions[session_id] = load_initial_state(session_id)
    return sessions[session_id]


def load_initial_state(session_id: str) -> AcademicState:
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    profile = {}
    calendar = {}
    tasks = {}

    try:
        with open(os.path.join(data_dir, "sample_profile.json")) as f:
            profile = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(data_dir, "sample_calendar.json")) as f:
            calendar = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(data_dir, "sample_tasks.json")) as f:
            tasks = json.load(f)
    except FileNotFoundError:
        pass

    return AcademicState(
        messages=[],
        profile=profile,
        calendar=calendar,
        tasks=tasks,
        results={},
        current_agent="idle",
        session_id=session_id,
    )


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    profile: Dict[str, Any]


class CalendarEventRequest(BaseModel):
    title: str
    start_datetime: str
    end_datetime: str
    description: Optional[str] = ""
    course: Optional[str] = ""


class TaskRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = ""
    priority: Optional[str] = "medium"
    course: Optional[str] = ""


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    completed: Optional[bool] = None


# ─── Chat Endpoint (SSE Streaming) ───────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    state = get_or_create_session(session_id)
    state["messages"] = state.get("messages", []) + [HumanMessage(content=request.message)]

    async def event_generator():
        yield {"event": "session", "data": json.dumps({"session_id": session_id})}

        full_response = ""
        active_agent = None
        # Track whether we received real streaming chunks per agent
        agent_streamed_chars: dict = {}

        try:
            async for event in graph.astream_events(state, version="v2"):
                event_type = event.get("event", "")
                node_name = event.get("name", "")

                if event_type == "on_chain_start" and node_name in ("coordinator", "planner", "notewriter", "advisor"):
                    active_agent = node_name
                    agent_streamed_chars[node_name] = 0
                    yield {
                        "event": "agent_start",
                        "data": json.dumps({"agent": node_name}),
                    }

                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        if active_agent and active_agent != "coordinator":
                            full_response += chunk.content
                            agent_streamed_chars[active_agent] = agent_streamed_chars.get(active_agent, 0) + len(chunk.content)
                            yield {
                                "event": "chunk",
                                "data": json.dumps({"content": chunk.content, "agent": active_agent}),
                            }

                elif event_type == "on_chain_end" and node_name in ("coordinator", "planner", "notewriter", "advisor"):
                    output = event.get("data", {}).get("output", {})
                    results = output.get("results", {}) if isinstance(output, dict) else {}

                    if node_name == "coordinator":
                        analysis = results.get("coordinator_analysis", {})
                        yield {
                            "event": "coordinator_done",
                            "data": json.dumps({
                                "required_agents": analysis.get("required_agents", []),
                                "reasoning": analysis.get("reasoning", ""),
                            }),
                        }

                    # Fallback: if no streaming chunks arrived, send full response as one chunk
                    elif agent_streamed_chars.get(node_name, 0) == 0:
                        response_key = f"{node_name}_response"
                        text = results.get(response_key, "")
                        if text:
                            full_response = text
                            yield {
                                "event": "chunk",
                                "data": json.dumps({"content": text, "agent": node_name}),
                            }

                    yield {
                        "event": "agent_end",
                        "data": json.dumps({"agent": node_name}),
                    }

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

        if full_response:
            from langchain_core.messages import AIMessage
            sessions[session_id]["messages"].append(AIMessage(content=full_response))

        yield {"event": "done", "data": json.dumps({"session_id": session_id})}

    return EventSourceResponse(event_generator())


# ─── Session ─────────────────────────────────────────────────────────────────

@app.post("/api/session")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = load_initial_state(session_id)
    return {"session_id": session_id}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[session_id]
    return {
        "session_id": session_id,
        "message_count": len(state.get("messages", [])),
        "has_profile": bool(state.get("profile")),
    }


# ─── Profile ─────────────────────────────────────────────────────────────────

@app.get("/api/profile/{session_id}")
async def get_profile(session_id: str):
    state = get_or_create_session(session_id)
    return state.get("profile", {})


@app.put("/api/profile/{session_id}")
async def update_profile(session_id: str, req: ProfileUpdateRequest):
    state = get_or_create_session(session_id)
    from state import dict_reducer
    sessions[session_id]["profile"] = dict_reducer(state.get("profile", {}), req.profile)
    return {"status": "updated", "profile": sessions[session_id]["profile"]}


# ─── Calendar ────────────────────────────────────────────────────────────────

@app.get("/api/calendar/{session_id}")
async def get_calendar(session_id: str):
    state = get_or_create_session(session_id)
    return state.get("calendar", {"events": []})


@app.post("/api/calendar/{session_id}/events")
async def add_calendar_event(session_id: str, req: CalendarEventRequest):
    state = get_or_create_session(session_id)
    event_id = str(uuid.uuid4())[:8]
    new_event = {
        "id": event_id,
        "title": req.title,
        "start": {"dateTime": req.start_datetime},
        "end": {"dateTime": req.end_datetime},
        "description": req.description,
        "course": req.course,
    }
    if "calendar" not in sessions[session_id] or not sessions[session_id]["calendar"]:
        sessions[session_id]["calendar"] = {"events": []}
    sessions[session_id]["calendar"].setdefault("events", []).append(new_event)
    return {"status": "created", "event": new_event}


@app.delete("/api/calendar/{session_id}/events/{event_id}")
async def delete_calendar_event(session_id: str, event_id: str):
    state = get_or_create_session(session_id)
    events = sessions[session_id].get("calendar", {}).get("events", [])
    sessions[session_id]["calendar"]["events"] = [e for e in events if e.get("id") != event_id]
    return {"status": "deleted"}


# ─── Tasks ────────────────────────────────────────────────────────────────────

@app.get("/api/tasks/{session_id}")
async def get_tasks(session_id: str):
    state = get_or_create_session(session_id)
    return state.get("tasks", {"tasks": []})


@app.post("/api/tasks/{session_id}")
async def create_task(session_id: str, req: TaskRequest):
    state = get_or_create_session(session_id)
    task_id = str(uuid.uuid4())[:8]
    new_task = {
        "id": task_id,
        "title": req.title,
        "description": req.description,
        "due_date": req.due_date,
        "priority": req.priority,
        "course": req.course,
        "completed": False,
    }
    if "tasks" not in sessions[session_id] or not sessions[session_id]["tasks"]:
        sessions[session_id]["tasks"] = {"tasks": []}
    sessions[session_id]["tasks"].setdefault("tasks", []).append(new_task)
    return {"status": "created", "task": new_task}


@app.put("/api/tasks/{session_id}/{task_id}")
async def update_task(session_id: str, task_id: str, req: TaskUpdateRequest):
    state = get_or_create_session(session_id)
    tasks = sessions[session_id].get("tasks", {}).get("tasks", [])
    for task in tasks:
        if task.get("id") == task_id:
            if req.title is not None:
                task["title"] = req.title
            if req.description is not None:
                task["description"] = req.description
            if req.due_date is not None:
                task["due_date"] = req.due_date
            if req.priority is not None:
                task["priority"] = req.priority
            if req.completed is not None:
                task["completed"] = req.completed
            return {"status": "updated", "task": task}
    raise HTTPException(status_code=404, detail="Task not found")


@app.delete("/api/tasks/{session_id}/{task_id}")
async def delete_task(session_id: str, task_id: str):
    state = get_or_create_session(session_id)
    tasks = sessions[session_id].get("tasks", {}).get("tasks", [])
    sessions[session_id]["tasks"]["tasks"] = [t for t in tasks if t.get("id") != task_id]
    return {"status": "deleted"}


# ─── History ──────────────────────────────────────────────────────────────────

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    state = get_or_create_session(session_id)
    messages = state.get("messages", [])
    return {
        "messages": [
            {
                "role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
                "content": m.content,
            }
            for m in messages
        ]
    }


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    if session_id in sessions:
        sessions[session_id]["messages"] = []
        sessions[session_id]["results"] = {}
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
