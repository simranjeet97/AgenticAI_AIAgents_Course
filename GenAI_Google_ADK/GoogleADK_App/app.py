import google.generativeai as genai
import streamlit as st
from agent import root_agent
import os
from dotenv import load_dotenv

genai.configure(api_key="")

# Configure page
st.set_page_config(page_title="📄 PDF Q&A Agent", layout="wide")

# --- Header Section ---
st.markdown("""
# 🤖 Gemini PDF Q&A Agent  
Ask questions about PDF documents — powered by **Gemini AI** and **Google ADK**.
""")
st.image("https://miro.medium.com/v2/resize:fit:1400/1*MU3ZjY0IMHdE0SCu57i5sA.gif", width=200)
# --- Main Layout: Two Columns ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    st.markdown("""
    - PDF is not stored — processed securely in-memory.
    - Works best with clean, text-based PDFs.
    """)

with col2:
    st.header("💬 Ask a Question")
    question = st.text_input("What would you like to know?", placeholder="e.g., What is the main conclusion of this document?")

    if uploaded_file and question:
        with st.spinner("💡 Thinking..."):
            try:
                pdf_bytes = uploaded_file.read()
                tool = root_agent.tools[0]
                result = tool(pdf_bytes=pdf_bytes, question=question)

                if result["status"] == "success":
                    st.success("✅ Answer")
                    st.markdown(f"**Question:** {question}")
                    st.markdown("---")
                    st.markdown(result["answer"])
                else:
                    st.error("⚠️ Could not process your question.")
                    st.code(result["error_message"])
            except Exception as e:
                st.error("🚨 Unexpected error")
                st.code(str(e))
    elif not uploaded_file and question:
        st.warning("📂 Please upload a PDF before asking.")
    elif uploaded_file and not question:
        st.info("✏️ Type your question in the box to get started.")

# --- Footer / Agent Info ---
st.markdown("---")
with st.expander("ℹ️ About This Agent"):
    st.markdown(f"""
    - **Agent Name**: `{root_agent.name}`  
    - **Model**: `{root_agent.model}`  
    - **Tools**: {[tool.__name__ for tool in root_agent.tools]}  
    - **Description**: {root_agent.description}

    Built using [Google ADK](https://google.github.io/adk-docs/) + Gemini.
    """)
