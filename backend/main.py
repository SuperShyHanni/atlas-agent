import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

import database as db
from auth import create_token, get_current_user_id, hash_password, verify_password
from graph import get_llm, graph
from memory_manager import (
    extract_and_store_memory,
    load_full_context,
    maybe_summarize,
    save_assistant_message,
    save_user_message,
)
from state import AcademicState, dict_reducer


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield


app = FastAPI(title="ATLAS Academic Agent API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_sample_data() -> Dict:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    result = {}
    for key, fname in (
        ("profile", "sample_profile.json"),
        ("calendar", "sample_calendar.json"),
        ("tasks", "sample_tasks.json"),
    ):
        try:
            with open(os.path.join(data_dir, fname)) as f:
                result[key] = json.load(f)
        except FileNotFoundError:
            result[key] = {}
    return result


async def _get_user_state_or_404(user_id: str) -> Dict:
    state = await db.get_user_state(user_id)
    if not state:
        raise HTTPException(status_code=404, detail="User state not found")
    return state


# ── Pydantic Models ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str


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


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    if await db.get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    await db.create_user(user_id, req.email, hash_password(req.password))

    sample = _load_sample_data()
    await db.upsert_user_state(
        user_id, sample["profile"], sample["calendar"], sample["tasks"]
    )

    token = create_token(user_id)
    return {"token": token, "user_id": user_id, "email": req.email}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = await db.get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(user["id"])
    return {"token": token, "user_id": user["id"], "email": user["email"]}


@app.get("/api/auth/me")
async def me(user_id: str = Depends(get_current_user_id)):
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ── Chat Endpoint (SSE Streaming) ─────────────────────────────────────────────

async def _stream_graph_run(graph_input, config, user_id, request_message, run_id):
    """Shared SSE generator for a fresh run (graph_input=state) or a resume
    (graph_input=None, same thread_id in config → LangGraph continues from the
    last checkpoint). On crash/disconnect the checkpoint persists, so the client
    can call /api/chat/resume/{run_id} to finish the run from where it stopped."""
    llm = get_llm()
    full_response = ""
    active_agent = None
    agent_streamed_chars: dict = {}

    try:
        async for event in graph.astream_events(graph_input, config=config, version="v2"):
            event_type = event.get("event", "")
            node_name = event.get("name", "")

            if event_type == "on_chain_start" and node_name in (
                "coordinator", "planner", "notewriter", "advisor"
            ):
                active_agent = node_name
                agent_streamed_chars[node_name] = 0
                yield {"event": "agent_start", "data": json.dumps({"agent": node_name})}

            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if active_agent and active_agent != "coordinator":
                        full_response += chunk.content
                        agent_streamed_chars[active_agent] = (
                            agent_streamed_chars.get(active_agent, 0) + len(chunk.content)
                        )
                        yield {
                            "event": "chunk",
                            "data": json.dumps({"content": chunk.content, "agent": active_agent}),
                        }

            elif event_type == "on_chain_end" and node_name in (
                "coordinator", "planner", "notewriter", "advisor"
            ):
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
                elif agent_streamed_chars.get(node_name, 0) == 0:
                    text = results.get(f"{node_name}_response", "")
                    if text:
                        full_response = text
                        yield {
                            "event": "chunk",
                            "data": json.dumps({"content": text, "agent": node_name}),
                        }

                yield {"event": "agent_end", "data": json.dumps({"agent": node_name})}

    except Exception as e:
        # Checkpoint is intact; surface run_id so the client can resume this run.
        yield {
            "event": "error",
            "data": json.dumps({"message": str(e), "run_id": run_id, "resumable": True}),
        }
        return

    if full_response:
        await save_assistant_message(user_id, full_response)
        await extract_and_store_memory(user_id, request_message, full_response, llm)
        await maybe_summarize(user_id, llm)

    yield {"event": "done", "data": json.dumps({"user_id": user_id, "run_id": run_id})}


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    user_state = await _get_user_state_or_404(user_id)
    await save_user_message(user_id, request.message)

    # L1+L2+L3: build full context (long-term recall + summary + recent msgs)
    context_messages = await load_full_context(user_id, request.message)
    context_messages.append(HumanMessage(content=request.message))

    # Per-run checkpoint thread — isolates resume state from the long-term memory
    # system, which already manages cross-turn context via load_full_context.
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{user_id}:{run_id}"}}

    state = AcademicState(
        messages=context_messages,
        profile=user_state["profile"],
        calendar=user_state["calendar"],
        tasks=user_state["tasks"],
        results={},
        current_agent="idle",
        session_id=user_id,
        run_id=run_id,
    )

    return EventSourceResponse(
        _stream_graph_run(state, config, user_id, request.message, run_id)
    )


@app.post("/api/chat/resume/{run_id}")
async def chat_resume(
    run_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Resume a previously crashed/disconnected run from its last checkpoint."""
    config = {"configurable": {"thread_id": f"{user_id}:{run_id}"}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.created_at:
        raise HTTPException(status_code=404, detail="No resumable run found for this id")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="Run already completed")

    # Recover the original request message for memory extraction after resume.
    rows = await db.get_all_messages(user_id)
    last_user = next((r["content"] for r in reversed(rows) if r["role"] == "user"), "")

    # input=None → LangGraph continues the existing thread from its last checkpoint.
    return EventSourceResponse(
        _stream_graph_run(None, config, user_id, last_user, run_id)
    )


# ── Profile ───────────────────────────────────────────────────────────────────

@app.get("/api/profile")
async def get_profile(user_id: str = Depends(get_current_user_id)):
    state = await _get_user_state_or_404(user_id)
    return state["profile"]


@app.put("/api/profile")
async def update_profile(
    req: ProfileUpdateRequest,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    merged = dict_reducer(state["profile"], req.profile)
    await db.update_state_field(user_id, "profile", merged)
    return {"status": "updated", "profile": merged}


# ── Calendar ──────────────────────────────────────────────────────────────────

@app.get("/api/calendar")
async def get_calendar(user_id: str = Depends(get_current_user_id)):
    state = await _get_user_state_or_404(user_id)
    return state["calendar"]


@app.post("/api/calendar/events")
async def add_calendar_event(
    req: CalendarEventRequest,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    calendar = state["calendar"]
    event_id = str(uuid.uuid4())[:8]
    new_event = {
        "id": event_id,
        "title": req.title,
        "start": {"dateTime": req.start_datetime},
        "end": {"dateTime": req.end_datetime},
        "description": req.description,
        "course": req.course,
    }
    calendar.setdefault("events", []).append(new_event)
    await db.update_state_field(user_id, "calendar", calendar)
    return {"status": "created", "event": new_event}


@app.delete("/api/calendar/events/{event_id}")
async def delete_calendar_event(
    event_id: str,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    calendar = state["calendar"]
    calendar["events"] = [e for e in calendar.get("events", []) if e.get("id") != event_id]
    await db.update_state_field(user_id, "calendar", calendar)
    return {"status": "deleted"}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/api/tasks")
async def get_tasks(user_id: str = Depends(get_current_user_id)):
    state = await _get_user_state_or_404(user_id)
    return state["tasks"]


@app.post("/api/tasks")
async def create_task(
    req: TaskRequest,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    tasks_data = state["tasks"]
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
    tasks_data.setdefault("tasks", []).append(new_task)
    await db.update_state_field(user_id, "tasks", tasks_data)
    return {"status": "created", "task": new_task}


@app.put("/api/tasks/{task_id}")
async def update_task(
    task_id: str,
    req: TaskUpdateRequest,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    tasks_data = state["tasks"]
    for task in tasks_data.get("tasks", []):
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
            await db.update_state_field(user_id, "tasks", tasks_data)
            return {"status": "updated", "task": task}
    raise HTTPException(status_code=404, detail="Task not found")


@app.delete("/api/tasks/{task_id}")
async def delete_task(
    task_id: str,
    user_id: str = Depends(get_current_user_id),
):
    state = await _get_user_state_or_404(user_id)
    tasks_data = state["tasks"]
    tasks_data["tasks"] = [t for t in tasks_data.get("tasks", []) if t.get("id") != task_id]
    await db.update_state_field(user_id, "tasks", tasks_data)
    return {"status": "deleted"}


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(user_id: str = Depends(get_current_user_id)):
    rows = await db.get_all_messages(user_id)
    return {"messages": [{"role": r["role"], "content": r["content"]} for r in rows]}


@app.delete("/api/history")
async def clear_history(user_id: str = Depends(get_current_user_id)):
    from memory_manager import clear_long_term_memory
    await db.clear_messages(user_id)
    await clear_long_term_memory(user_id)
    return {"status": "cleared"}


# ── Documents (RAG) ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    filename: str


@app.post("/api/documents/ingest")
async def ingest_document(
    req: IngestRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Chunk + embed a file from the student's FILES_DIR into the RAG store."""
    from rag import ingest_file
    result = ingest_file(user_id, req.filename)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/documents")
async def list_documents(user_id: str = Depends(get_current_user_id)):
    from vector_store import list_document_sources
    return {"sources": list_document_sources(user_id)}


@app.delete("/api/documents")
async def delete_documents(
    source: str = "",
    user_id: str = Depends(get_current_user_id),
):
    from vector_store import delete_user_documents
    delete_user_documents(user_id, source=source)
    return {"status": "deleted", "source": source or "all"}


# ── Tool Audit (RBAC trace) ───────────────────────────────────────────────────

@app.get("/api/audit/tools")
async def get_tool_audit(
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
):
    """Return this user's tool-call audit trail (allowed/denied/error), newest first."""
    rows = await db.get_tool_audit(user_id, limit=min(limit, 500))
    return {"audit": rows}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
