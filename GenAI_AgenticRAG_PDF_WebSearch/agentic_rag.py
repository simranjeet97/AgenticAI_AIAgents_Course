import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import google.generativeai as genai
import bs4
from agno.vectordb.chroma import ChromaDb
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools

import hnswlib

# Monkey-patch: set file_handle_count to a numeric value rather than a function
if not hasattr(hnswlib.Index, "file_handle_count"):
    hnswlib.Index.file_handle_count = 0

class GeminiEmbedder(Embeddings):
    def __init__(self, model_name="models/embedding-001"):
        genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

    def get_embedding(self, text: str) -> List[float]:
        return self.embed_query(text)

# Constants
COLLECTION_NAME = "gemini-thinking-agent-agno"

# Streamlit App Initialization
st.title("🤔 Agentic RAG with Gemini Flash Thinking and Agno")

# Session State Initialization
default_session_values = {
    'google_api_key': "",
    'vector_store': None,
    'processed_documents': [],
    'history': [],
    'exa_api_key': "2c9220da-94f0-45a7-bbf9-ffa8a1a4ae9c",
    'use_web_search': False,
    'force_web_search': False,
    'similarity_threshold': 0.7
}
for key, value in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
st.session_state.google_api_key = google_api_key

EMBEDDING_MODEL = GeminiEmbedder()

def init_chroma():
    """Initializes ChromaDB and ensures the collection exists."""
    chroma = ChromaDb(
        collection=COLLECTION_NAME,
        path='./chroma_db',
        embedder=EMBEDDING_MODEL,
        persistent_client=True
    )
    try:
        chroma.client.get_collection(name=COLLECTION_NAME)
    except Exception:
        chroma.create()
    return chroma

def process_pdf(file) -> List:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []

def process_web(url: str) -> List:
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []

def create_vector_store(collection, texts):
    try:
        embeddings = GeminiEmbedder()
        with st.spinner('📤 Uploading documents to Chroma...'):
            for text in texts:
                embedding = embeddings.embed_query(text.page_content)
                collection.add(
                    documents=[text.page_content],
                    metadatas=[text.metadata],
                    ids=[str(hash(text.page_content))],
                    embeddings=[embedding]
                )
            st.success("✅ Documents stored successfully!")
            return collection
    except Exception as e:
        st.error(f"🔴 Chroma vector store error: {str(e)}")
        return None

def get_query_rewriter_agent() -> Agent:
    return Agent(
        name="Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions="""You are an expert at reformulating questions to be more precise and detailed. 
        Rewrite questions to be more specific and search-friendly.
        """,
        show_tool_calls=False,
        markdown=True,
    )

def get_web_search_agent() -> Agent:
    return Agent(
        name="Web Search Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[ExaTools(api_key=st.session_state.exa_api_key, num_results=5)],
        instructions="""Search the web for relevant information and summarize findings with sources."""
    )

def get_rag_agent() -> Agent:
    return Agent(
        name="Gemini RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""
        When given document context, focus on relevant details and cite directly.
        When given web search results, state clearly they are from the web.
        Keep answers clear, concise, and accurate.
        """
    )

def retrieve_documents(prompt, client, COLLECTION_NAME):
    results = client.search(query=prompt, limit=5)
    has_docs = len(results) > 0
    return results, has_docs

# Main Application Flow
if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    genai.configure(api_key=st.session_state.google_api_key)

    client = init_chroma()
    collection_chroma = client.client.get_collection(name=COLLECTION_NAME)

    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts and collection_chroma:
                    collection_chroma = create_vector_store(collection_chroma, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner('Processing URL...'):
                texts = process_web(web_url)
                if texts and collection_chroma:
                    collection_chroma = create_vector_store(collection_chroma, texts)
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ Added URL: {web_url}")

    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            icon = "📄" if source.endswith('.pdf') else "🌐"
            st.sidebar.text(f"{icon} {source}")

    chat_col, toggle_col = st.columns([0.9, 0.1])

    with chat_col:
        prompt = st.chat_input("Ask about your documents...")

    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("🤔 Accessing Query..."):
            try:
                with st.spinner("📄 Retrieving documents..."):
                    results, has_docs = retrieve_documents(prompt, client, COLLECTION_NAME)

                final_answer = None

                if has_docs and not st.session_state.force_web_search:
                    rag_agent = get_rag_agent()
                    context = "\n\n".join([doc['document'] for doc in results['documents']])
                    final_answer = rag_agent.run(f"{prompt}\n\nContext:\n{context}").content
                else:
                    print("Going in Web Search")
                    web_search_agent = get_web_search_agent()
                    final_answer = web_search_agent.run(prompt).content

                st.session_state.history.append({"role": "assistant", "content": final_answer})
                with st.chat_message("assistant"):
                    st.markdown(final_answer)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")