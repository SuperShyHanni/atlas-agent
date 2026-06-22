# ATLAS — Academic Task & Learning Agent System

A full-stack, multi-agent AI assistant for students, built with **LangGraph + FastAPI** (backend) and **React + TypeScript** (frontend).

A Coordinator performs intent recognition and routes each request to one of three specialist agents (Planner / NoteWriter / Advisor), each running a **ReAct** tool-calling loop. Beyond a basic agent demo, ATLAS is engineered as a production-style system: a **three-layer memory architecture**, **hybrid retrieval RAG** with an offline evaluation harness, **tool-level RBAC + audit + idempotency**, a **self-built SQLite checkpointer** for crash/interrupt recovery, and **`ContextVar`-based session isolation** for safe concurrency.

> Originally inspired by the ATLAS notebook by NirDiamant, then re-architected from scratch into a stateful, fault-tolerant web application.

---

## ✨ Engineering Highlights

| Area | What it does |
|------|--------------|
| 🧭 **Multi-agent orchestration** | Coordinator routes to specialist agents over a LangGraph `StateGraph`; declarative conditional edges keep routing testable without an LLM. |
| 🔁 **ReAct loop** | Each agent reasons → calls tools → observes results → reasons again, terminating when no more tool calls are emitted. |
| 🧠 **Three-layer memory (L1/L2/L3)** | Working memory (recent turns) + LLM-summarized short-term memory + long-term facts embedded in ChromaDB, recalled by similarity. Keeps prompt size roughly constant on long conversations and persists facts across sessions. |
| 🔎 **Hybrid RAG** | Vector + BM25 fused via **weighted Reciprocal Rank Fusion**; chunk size and fusion weights tuned with an offline eval set (AnswerHit@k / MRR). Answers cite their sources (`[source p.X]`). |
| 🛡️ **Tool RBAC + audit + idempotency** | Every tool call is authorized per agent, written to an immutable audit log, and deduplicated within a run via an `(run_id, tool, args)` cache. |
| 💾 **Crash/interrupt recovery** | A self-implemented SQLite `BaseCheckpointSaver` snapshots full state after every node; a disconnected run resumes from the last completed node via `/api/chat/resume`. |
| 🔒 **Safe concurrency** | `ContextVar` isolates per-request state into each asyncio task, fixing cross-session leakage that a module-global would cause. |
| 🔐 **Auth** | JWT + bcrypt; all data is scoped per user. |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       React + TypeScript                      │
│      Sidebar (Profile / Calendar / Tasks / Docs) │ Chat       │
└───────────────────────────┬──────────────────────────────────┘
                            │  SSE streaming (POST /api/chat/stream)
┌───────────────────────────▼──────────────────────────────────┐
│                          FastAPI                              │
│                                                              │
│  Auth (JWT/bcrypt) · REST (profile/calendar/tasks/docs)      │
│                                                              │
│  ┌───────────────────── LangGraph StateGraph ─────────────┐  │
│  │  START → Coordinator ──(conditional routing)──┐         │  │
│  │            ┌──────────────────────────────────▼──────┐  │  │
│  │            │  Planner 📅  │ NoteWriter ✍️ │ Advisor 💡 │  │  │
│  │            └───────────────┬──────────────────────────┘  │  │
│  │                    has tool_calls?                       │  │
│  │                  ┌─────────┴─────────┐                   │  │
│  │             RBAC tools node         END                  │  │
│  │            (authorize+audit+        (final answer)       │  │
│  │             idempotency cache)                           │  │
│  │                  └──── back to calling agent (ReAct) ────┘  │
│  │                                                          │  │
│  │  Checkpointer: snapshots full state after each node ─────┘  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
│  Memory: L1 (SQLite recent) · L2 (LLM summary) · L3 (ChromaDB)│
│  Retrieval: vector + BM25 → weighted RRF                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                OpenAI-compatible LLM endpoint
              (default: Volcano Ark; any compatible API works)
```

---

## Agents

| Agent | Role | Routed when the request is about… |
|-------|------|-----------------------------------|
| **Coordinator** | Intent recognition; routes to a specialist (structured JSON with robust fallback) | every message |
| **Planner** 📅 | schedules, deadlines, study sessions, time management | scheduling / calendar / planning |
| **NoteWriter** ✍️ | notes, summaries, study guides, flashcards | content / notes / summaries |
| **Advisor** 💡 | learning strategies, performance analysis, motivation | advice / guidance / strategy |

**Tools** (gated by RBAC): profile, calendar, tasks, filesystem, Notion, and `search_documents` (RAG over the student's own uploaded materials).

---

## Core Systems

### 🧠 Three-Layer Memory
- **L1 — Working memory**: the most recent turns, kept verbatim in context.
- **L2 — Short-term memory**: older history compressed into a rolling LLM summary (incremental — the prior summary is folded back in so nothing silently drops).
- **L3 — Long-term memory**: durable, student-stated facts embedded in ChromaDB and recalled by semantic similarity, scoped per user.
- **Anti-hallucination**: extraction is constrained to the student's own wording (no inference / no paraphrase), recall injects a "cite verbatim, ask instead of guessing" instruction, and agents are told never to invent numbers.

### 🔎 Hybrid Retrieval RAG
- Dense vector search **+** BM25 lexical search, fused with **weighted RRF** (`score = Σ wᵢ / (c + rankᵢ + 1)`, `c=60`, vector weight 0.8).
- Chunking (200/40, sentence-boundary aware) and fusion weights were **swept on an offline eval set** — see `backend/eval/`.
- Documents are ingested per user; answers cite `[source p.X]`.

### 🛡️ Reliability & Safety
- **RBAC + audit**: a custom guarded tool node authorizes each call per agent and logs every call (allowed/denied/error) with latency.
- **Idempotency**: identical `(run_id, tool, args)` calls execute once per run.
- **Checkpointer**: a from-scratch SQLite `BaseCheckpointSaver` (the official package isn't available offline) persists every super-step for resume-from-last-node recovery.
- **Session isolation**: `ContextVar` scopes injected request state to each asyncio task.

---

## Tech Stack

**Backend** — Python, LangGraph, LangChain, FastAPI, SSE, ChromaDB, SQLite (aiosqlite), PyJWT, bcrypt
**Frontend** — React 18, TypeScript, Vite, Tailwind CSS, react-markdown
**LLM** — any OpenAI-compatible endpoint via `langchain-openai` (default base URL: Volcano Ark)

---

## Project Structure

```
atlas-agent/
├── backend/
│   ├── main.py            # FastAPI app: auth, chat streaming/resume, REST, docs, audit
│   ├── graph.py           # LangGraph wiring: nodes, conditional edges, ReAct loop
│   ├── state.py           # AcademicState (TypedDict) + reducers
│   ├── memory_manager.py  # L1/L2/L3 memory: summarize, extract, recall, assemble context
│   ├── rag.py             # document ingestion + chunking
│   ├── vector_store.py    # ChromaDB + BM25 + weighted RRF hybrid search
│   ├── rbac.py            # guarded tool node: authorize + audit + idempotency
│   ├── checkpointer.py    # self-built SQLite checkpointer
│   ├── agents/            # coordinator, planner, notewriter, advisor
│   ├── tools/             # profile, calendar, tasks, filesystem, notion, rag + ContextVar
│   └── eval/              # offline RAG evaluation harness (goldset + sweeps)
└── frontend/              # React + TypeScript + Vite UI
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- An OpenAI-compatible LLM API key (any provider; default config targets Volcano Ark)

### Backend
```bash
cd backend
cp .env.example .env        # fill in the values below
pip install -r requirements.txt
python main.py              # http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for your LLM endpoint | **required** |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL | `https://ark.cn-beijing.volces.com/api/v3` |
| `LLM_MODEL` | Model / endpoint id | **required** |
| `JWT_SECRET_KEY` | Secret for signing auth tokens | dev fallback |

---

## Evaluation

The retrieval pipeline ships with a reproducible, offline (no-network) eval harness:

```bash
cd backend
python eval/evaluate_rag.py          # compare vector / bm25 / hybrid on the gold set
python eval/evaluate_rag.py sweep    # tune chunk size + fusion weight
```

Metrics: **SourceRecall@k**, **AnswerHit@k** (retrieval precision), **AnswerMRR** (ranking quality). Ground truth is the chunk containing the gold keyword, so results are deterministic and reproducible. (The current gold set is small — treated as a directional sanity check, not a significance-grade benchmark.)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` · `/api/auth/login` | Auth (JWT) |
| GET | `/api/auth/me` | Current user |
| POST | `/api/chat/stream` | Streaming chat (SSE) |
| POST | `/api/chat/resume/{run_id}` | Resume an interrupted run from its last checkpoint |
| GET/PUT | `/api/profile` | Student profile |
| GET/POST/DELETE | `/api/calendar` · `/api/calendar/events` | Calendar events |
| GET/POST/PUT/DELETE | `/api/tasks` | Task management |
| POST/GET/DELETE | `/api/documents` · `/api/documents/ingest` | RAG document ingestion |
| GET/DELETE | `/api/history` | Chat history |
| GET | `/api/audit/tools` | Tool-call audit log |
| GET | `/api/health` | Health check |
