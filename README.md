# ATLAS — Academic Task & Learning Agent System

A full-stack AI agent application built with **Python + LangGraph + FastAPI** (backend) and **React + TypeScript** (frontend).

Based on the ATLAS notebook by NirDiamant, re-architected as a production web application with real-time streaming.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    React Frontend                   │
│  Sidebar (Profile/Calendar/Tasks) │ Chat │ Agents  │
└────────────────────┬────────────────────────────────┘
                     │ SSE Streaming (POST /api/chat/stream)
┌────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                    │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │           LangGraph Workflow                │   │
│  │                                             │   │
│  │  START → Coordinator → (routing) →          │   │
│  │          ┌──────────────────────┐           │   │
│  │          │  Planner Agent  📅  │           │   │
│  │          │  NoteWriter Agent ✍️│           │   │
│  │          │  Advisor Agent  💡 │           │   │
│  │          └──────────────────────┘           │   │
│  │                      → END                  │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  REST APIs: /api/profile, /api/calendar, /api/tasks │
└─────────────────────────────────────────────────────┘
                     │
              Claude claude-sonnet-4-6
              (Anthropic API)
```

## Agents

| Agent | Role | Triggers |
|-------|------|----------|
| **Coordinator** | Analyzes request, routes to specialists | Every message |
| **Planner** 📅 | Study schedules, deadlines, time management | Schedule/calendar/planning requests |
| **NoteWriter** ✍️ | Study notes, summaries, flashcards, study guides | Content/notes/summary requests |
| **Advisor** 💡 | Academic strategies, learning tips, motivation | Advice/guidance/strategy requests |

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)
- Anthropic API key

### Backend

```bash
cd backend
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

pip install -r requirements.txt
python main.py
# Server runs at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# App runs at http://localhost:5173
```

## Features

- **Real-time streaming** — Watch agents respond token by token via SSE
- **Agent activity panel** — See which agents are active in real time
- **Student profile** — Personalize responses based on your learning style
- **Calendar management** — Track upcoming events and deadlines
- **Task tracker** — Manage assignments with priorities and due dates
- **Persistent sessions** — Conversation history maintained per session
- **Markdown rendering** — Beautifully formatted agent responses

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `MODEL_NAME` | Claude model to use | `claude-sonnet-4-6` |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/stream` | Streaming chat (SSE) |
| POST | `/api/session` | Create new session |
| GET/PUT | `/api/profile/{session_id}` | Student profile |
| GET/POST | `/api/calendar/{session_id}` | Calendar events |
| GET/POST/PUT/DELETE | `/api/tasks/{session_id}` | Task management |
| GET/DELETE | `/api/history/{session_id}` | Chat history |
| GET | `/api/health` | Health check |
