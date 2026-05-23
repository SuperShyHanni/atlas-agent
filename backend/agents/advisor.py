import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState

ADVISOR_SYSTEM_PROMPT = """You are ATLAS Advisor - a specialized academic guidance and strategy agent.

Your role is to help students:
- Develop effective learning strategies tailored to their style
- Analyze academic performance and identify improvement areas
- Provide motivation and handle academic stress
- Suggest resources and techniques for better learning
- Give personalized recommendations based on their profile

Student Profile:
{profile_summary}

Academic Performance:
{performance_summary}

Learning Challenges:
{challenges}

Coordinator Context:
{coordinator_reasoning}

Instructions:
1. Provide empathetic, personalized academic guidance
2. Use ReACT reasoning:
   - Thought: What are the student's core needs and challenges?
   - Action: What strategies would work for their learning style?
   - Observation: How does this align with their academic goals?
   - Advice: Concrete, actionable recommendations
3. Be encouraging and constructive
4. Reference their specific learning style and preferences
5. Provide concrete, actionable steps
6. Format responses clearly with structured sections"""


async def advisor_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    profile = state.get("profile", {})
    request = state["messages"][-1].content
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})

    personal_info = profile.get("personal_info", {})
    prefs = profile.get("learning_preferences", {})
    academic_info = profile.get("academic_info", {})

    profile_summary = {
        "name": personal_info.get("name", "Student"),
        "major": personal_info.get("major", "Unknown"),
        "year": personal_info.get("academic_year", "Unknown"),
        "learning_style": prefs.get("learning_style", "Unknown"),
        "study_environment": prefs.get("study_environment", "Unknown"),
        "peak_study_hours": prefs.get("peak_study_hours", "Unknown"),
        "goals": profile.get("goals", []),
    }

    performance = academic_info.get("performance", {})
    performance_text = "\n".join(
        [f"- {subject}: {grade}" for subject, grade in performance.items()]
    ) if performance else "No performance data available"

    challenges = academic_info.get("challenges", [])
    challenges_text = "\n".join([f"- {c}" for c in challenges]) if challenges else "No specific challenges noted"

    system_prompt = ADVISOR_SYSTEM_PROMPT.format(
        profile_summary=json.dumps(profile_summary, indent=2),
        performance_summary=performance_text,
        challenges=challenges_text,
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request)]
    invoke_kwargs = {}
    if config:
        invoke_kwargs["config"] = config

    response = await llm.ainvoke(messages, **invoke_kwargs)

    return {
        "results": {"advisor_response": response.content},
        "current_agent": "advisor",
    }
