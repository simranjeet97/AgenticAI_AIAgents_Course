import os
os.environ['OPENAI_API_KEY'] = ""  # Omit your key in public code

import asyncio
import streamlit as st
from datetime import datetime
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List
from agents import (
    Agent, Runner, WebSearchTool, function_tool, handoff,
    OpenAIChatCompletionsModel,
    set_default_openai_api, set_default_openai_client, set_tracing_disabled
)

# ---------------- SETUP ----------------
openai_client = AsyncOpenAI()
set_default_openai_api("chat_completions")
set_default_openai_client(openai_client)
set_tracing_disabled(True)

# ---------------- MODELS ----------------
class CodingPlan(BaseModel):
    problem: str
    subtasks: List[str]
    tech_stack: List[str]

class CodeOutput(BaseModel):
    filename: str
    code: str
    explanation: str

# ---------------- AGENTS ----------------
# Triage / Planning Agent
planner_agent = Agent(
    name="Planner Agent",
    instructions="""
You're a software planner. Given a requirement, break it into components:
1. Identify key features.
2. Suggest relevant tech stack/tools.

Respond in JSON:
{
  "problem": "<problem>",
  "subtasks": ["task1", "task2"],
  "tech_stack": ["Python", "Flask", ...]
}
""",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
    output_type=CodingPlan,
)

# Coder Agent
coder_agent = Agent(
    name="Code Generator",
    instructions="""
Given a task and tech stack, generate clear code and explain it.

Respond in this format:
{
  "filename": "file.py",
  "code": "...",
  "explanation": "..."
}
""",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
    output_type=CodeOutput,
)

# Reviewer Agent
reviewer_agent = Agent(
    name="Code Reviewer",
    instructions="Review code and suggest improvements, optimizations, or warnings.",
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="💻 SDE Coding Assistant", layout="wide")
st.title("💻 SDE Coding Assistant")
st.markdown("Streamline your development with planning, coding, and reviewing agents.")

# ---------------- SESSION STATE ----------------
if "code_plan" not in st.session_state:
    st.session_state.code_plan = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = None
if "code_review" not in st.session_state:
    st.session_state.code_review = None

# ---------------- INPUT ----------------
with st.sidebar:
    st.header("📥 Problem Input")
    problem_input = st.text_area("Describe your coding problem or feature request:")
    if st.button("Plan & Build", disabled=not problem_input):
        st.session_state.generated_code = None
        st.session_state.code_review = None

        async def run_planner():
            with st.spinner("🧠 Thinking and planning..."):
                plan_result = await Runner.run(planner_agent, problem_input)
                st.session_state.code_plan = plan_result.final_output

        asyncio.run(run_planner())

# ---------------- DISPLAY PLAN ----------------
if st.session_state.code_plan:
    plan = st.session_state.code_plan
    st.markdown(f"### 📋 Plan for: `{plan.problem}`")
    st.markdown("**🧩 Subtasks:**")
    for s in plan.subtasks:
        st.markdown(f"- {s}")
    st.markdown("**🧪 Suggested Tech Stack:**")
    st.markdown(", ".join(plan.tech_stack))

    if plan.subtasks:
        selected_task = st.selectbox("Select a subtask to implement:", plan.subtasks)

        if st.button("Generate Code for Task"):
            async def run_coder():
                with st.spinner("🛠️ Generating code..."):
                    input_text = f"Task: {selected_task}\nTech stack: {plan.tech_stack}"
                    code_result = await Runner.run(coder_agent, input_text)
                    st.session_state.generated_code = code_result.final_output

            asyncio.run(run_coder())
    else:
        st.warning("⚠️ No subtasks were returned by the planner agent.")

# ---------------- DISPLAY CODE ----------------
if st.session_state.generated_code:
    result = st.session_state.generated_code
    st.markdown(f"### 🧾 Generated File: `{result.filename}`")
    st.code(result.code, language=result.filename.split('.')[-1])
    st.markdown(f"🧠 **Explanation:** {result.explanation}")

    if st.button("🔍 Review Code"):
        async def run_reviewer():
            with st.spinner("🔍 Reviewing code..."):
                review_result = await Runner.run(reviewer_agent, result.code)
                st.session_state.code_review = review_result.final_output

        asyncio.run(run_reviewer())

# ---------------- DISPLAY REVIEW ----------------
if st.session_state.code_review:
    st.markdown("### 🛠️ Review Suggestions")
    st.markdown(st.session_state.code_review)