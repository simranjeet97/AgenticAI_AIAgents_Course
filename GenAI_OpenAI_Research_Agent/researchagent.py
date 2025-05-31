import os
os.environ['OPENAI_API_KEY'] = ""  # Omit your key in public code

import asyncio
import streamlit as st
from datetime import datetime
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

# ---------- INIT SESSION STATE FIRST ----------
for key, default in {
    "collected_facts": [],
    "research_started": False,
    "research_done": False,
    "report_result": None,
    "research_progress": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- SETUP ----------
openai_client = AsyncOpenAI()
set_default_openai_api("chat_completions")
set_default_openai_client(openai_client)
set_tracing_disabled(True)

# ---------- CUSTOM TOOL ----------
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    for existing in st.session_state.collected_facts:
        if existing["fact"] == fact:
            return "⚠️ Fact already saved."

    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "Not specified",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    return f"✅ Fact saved: {fact}"

# ---------- AGENTS ----------
query_rewriting_agent = Agent(
    name="Query Rewriting Agent",
    instructions="Given a topic, generate 5–8 diverse and precise search queries.\nRespond as a bullet list.",
    model=OpenAIChatCompletionsModel(model="gpt-4-turbo-2024-04-09", openai_client=openai_client),
)

research_agent = Agent(
    name="Research Agent",
    instructions="""
For each query:
1. Search the web.
2. Save key insights with save_important_fact().
3. Format output in Markdown.
""",
    model=OpenAIChatCompletionsModel(model="gpt-4-turbo-2024-04-09", openai_client=openai_client),
    tools=[WebSearchTool(), save_important_fact],
)

class ResearchReport(BaseModel):
    title: str
    outline: list[str]
    report: str
    sources: list[str]
    word_count: int

editor_agent = Agent(
    name="Editor Agent",
    instructions="Compile a structured research report.",
    model=OpenAIChatCompletionsModel(model="gpt-4o-2024-08-06", openai_client=openai_client),
    output_type=ResearchReport,
)

class ResearchPlan(BaseModel):
    topic: str
    search_queries: list[str]
    focus_areas: list[str]

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
Generate search queries and focus areas from the topic. Return JSON.
""",
    handoffs=[handoff(query_rewriting_agent), handoff(research_agent), handoff(editor_agent)],
    model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=openai_client),
    output_type=ResearchPlan,
)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.5em; font-weight: 700; color: #4CAF50; }
    .subtitle { font-size: 1.2em; color: #666; margin-top: -10px; }
    .fact-box { background-color: #f9f9f9; border-left: 5px solid #4CAF50; padding: 10px; margin: 10px 0; }
    .source-link { color: #1f77b4; text-decoration: none; font-weight: bold; }
</style>
<div class="main-title">🔬 AI Research Assistant</div>
<div class="subtitle">Conduct in-depth research using smart agents and web search</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📝 Research Input")
    user_topic = st.text_input("Enter a topic for research:")
    start_button = st.button("Start Research", type="primary", disabled=not user_topic)
    st.markdown("---")
    st.markdown("🔔 **Status**")
    st.progress(st.session_state.research_progress)
    st.caption("💡 Tip: Be specific with your topic for better results!")

# ---------- ASYNC RESEARCH FUNCTION ----------
async def run_research(topic: str):
    st.session_state.research_started = True
    st.session_state.research_done = False
    st.session_state.collected_facts = []
    st.session_state.report_result = None
    st.session_state.research_progress = 0

    st.markdown("### 🧠 Generating Research Plan...")
    plan_result = await Runner.run(triage_agent, f"Create a research plan for: {topic}")
    research_plan = plan_result.final_output

    st.session_state.research_progress = 25
    with st.expander("📋 Research Plan", expanded=True):
        st.markdown(f"**🔎 Topic:** `{research_plan.topic}`")
        st.markdown("<ul>" + "".join([f"<li>🔍 {q}</li>" for q in research_plan.search_queries]) + "</ul>", unsafe_allow_html=True)
        st.markdown("<ul>" + "".join([f"<li>🎯 {a}</li>" for a in research_plan.focus_areas]) + "</ul>", unsafe_allow_html=True)

    st.markdown("### 🔍 Conducting Research...")
    await asyncio.sleep(1.5)
    st.session_state.research_progress = 50

    with st.expander("📚 Collected Facts", expanded=True):
        if st.session_state.collected_facts:
            for fact in st.session_state.collected_facts:
                st.markdown(f"""
                <div class="fact-box">
                    <strong>🧠 {fact['fact']}</strong><br>
                    📌 Source: <a class="source-link">{fact['source']}</a><br>
                    🕒 {fact['timestamp']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No facts collected yet.")

    st.markdown("### 📝 Compiling Research Report...")
    editor_result = await Runner.run(editor_agent, plan_result.to_input_list())
    st.session_state.report_result = editor_result.final_output
    st.session_state.research_done = True
    st.session_state.research_progress = 100
    st.sidebar.progress(100)
    st.success("✅ Research Complete!")

# ---------- RUN RESEARCH ----------
if start_button:
    with st.spinner(f"Running research for: {user_topic}"):
        asyncio.run(run_research(user_topic))

# ---------- DISPLAY REPORT ----------
if st.session_state.research_done and st.session_state.report_result:
    report = st.session_state.report_result
    st.markdown(f"<h2>📖 {report.title}</h2>", unsafe_allow_html=True)
    st.success(f"📝 Word Count: {report.word_count}")

    with st.expander("🧾 Outline", expanded=True):
        for item in report.outline:
            st.markdown(f"- ✅ {item}")

    st.markdown("### 📄 Report Content")
    st.markdown(report.report)

    with st.expander("🔗 Sources", expanded=True):
        for src in report.sources:
            st.markdown(f"- 🔗 {src}")

    st.download_button(
        label="⬇️ Download Report",
        data=report.report,
        file_name=f"{report.title.replace(' ', '_')}.md",
        mime="text/markdown"
    )