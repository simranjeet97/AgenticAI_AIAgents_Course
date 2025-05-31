import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from composio_agno import Action, ComposioToolSet
from markdown2 import markdown
from fpdf import FPDF
from fpdf.html import HTML2FPDF, HTMLMixin
import os
import re
import html  # for unescape monkey patch

# Patch fpdf’s HTML2FPDF to add unescape
HTML2FPDF.unescape = staticmethod(html.unescape)

# Setup API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
composio_api_key = "3dms9ab2w9muwy6pa72l3a"

# Agent setup
composio_toolset = ComposioToolSet(api_key=composio_api_key)
search_tool = composio_toolset.get_tools(actions=[Action.COMPOSIO_SEARCH_DUCK_DUCK_GO_SEARCH])

# Improved agent setup with more precise instructions
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[search_tool],
    instructions="""
    You are an AI assistant tasked with providing detailed, accurate, and insightful information on a given topic. 
    Please ensure the following:
    1. Conduct a web search using the 'COMPOSIO_SEARCH_DUCK_DUCK_GO_SEARCH' tool.
    2. Provide a summary of the topic, including key facts, statistics, examples, and relevant details.
    3. Ensure accuracy by cross-referencing multiple reliable sources.
    4. Present the information in a markdown format with clear headings, bullet points, and links where applicable.
    5. Fact-check the details by searching online.
    6. Avoid redundancy and provide the most up-to-date and reliable information.
    """,
    show_tool_calls=True,
)

class PDF(FPDF, HTMLMixin):
    pass

# Set up Streamlit UI
st.title("🔎 AI Web Search with PDF Export")
query = st.text_input("Enter a topic to search:", "Re-Development Schemes in DCPR 2034")

if "search_result_md" not in st.session_state:
    st.session_state.search_result_md = ""

if st.button("Search"):
    with st.spinner("Searching..."):
        try:
            # Call the agent to search and process the topic
            result = agent.run(f"Search online and explain this topic in markdown:\n{query}", force_tool_calls=True)
            st.session_state.search_result_md = result.content.strip()
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.session_state.search_result_md:
    # Display the markdown result in the app
    st.markdown(st.session_state.search_result_md, unsafe_allow_html=True)