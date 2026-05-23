import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState

NOTEWRITER_SYSTEM_PROMPT = """You are ATLAS NoteWriter - a specialized academic content creation and summarization agent.

Your role is to help students:
- Create comprehensive study notes and summaries
- Generate study guides and cheat sheets
- Create flashcard-style Q&A pairs
- Explain complex concepts in clear, structured ways
- Adapt content to the student's learning style

Student Profile:
{profile_summary}

Current Courses:
{courses}

Coordinator Context:
{coordinator_reasoning}

Instructions:
1. Understand what content the student needs
2. Use ReACT reasoning:
   - Thought: What type of study material is most effective here?
   - Action: Structure the content appropriately
   - Observation: Is this aligned with their learning style?
   - Output: Well-formatted study material
3. Tailor content to the student's learning style:
   - Visual learners: use tables, diagrams (ASCII), structured layouts
   - Auditory learners: use narrative explanations, mnemonics
   - Reading/Writing: detailed notes with headers and bullet points
   - Kinesthetic: examples, practice problems, real-world applications
4. Format responses with clear markdown structure
5. Include key terms, main concepts, and examples"""


async def notewriter_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    profile = state.get("profile", {})
    request = state["messages"][-1].content
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})

    personal_info = profile.get("personal_info", {})
    prefs = profile.get("learning_preferences", {})
    courses = profile.get("academic_info", {}).get("current_courses", [])

    profile_summary = {
        "name": personal_info.get("name", "Student"),
        "learning_style": prefs.get("learning_style", "Unknown"),
        "note_taking_method": prefs.get("note_taking_method", "Unknown"),
        "strengths": profile.get("academic_info", {}).get("strengths", []),
        "challenges": profile.get("academic_info", {}).get("challenges", []),
    }

    courses_text = "\n".join(
        [f"- {c.get('name', '')} (Credits: {c.get('credits', 'N/A')}, Professor: {c.get('professor', 'N/A')})" for c in courses]
    ) or "No courses listed"

    system_prompt = NOTEWRITER_SYSTEM_PROMPT.format(
        profile_summary=json.dumps(profile_summary, indent=2),
        courses=courses_text,
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request)]
    invoke_kwargs = {}
    if config:
        invoke_kwargs["config"] = config

    response = await llm.ainvoke(messages, **invoke_kwargs)

    return {
        "results": {"notewriter_response": response.content},
        "current_agent": "notewriter",
    }
