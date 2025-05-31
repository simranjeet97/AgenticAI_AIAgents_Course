import os
import requests
import json
import streamlit as st
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from firecrawl import FirecrawlApp
from composio_agno import Action, ComposioToolSet
from typing import List, Dict
from pydantic import BaseModel, Field
from agno.models.google import Gemini
import re

# Set Google Gemini API key
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit UI
st.set_page_config(page_title="Google Search Scraper", layout="wide")
st.title("🔍 Google Job Search Scraper & Data Exporter")

# User input
query = st.text_input("Enter search query:", "Hiring Data Scientist")
num_results = st.slider("Number of results:", 1, 10, 2)
firecrawl_api_key = st.text_input("Enter Firecrawl API Key:", type="password")

# Initialize session state for caching results
if "formatted_data" not in st.session_state:
    st.session_state.formatted_data = []
if "urls" not in st.session_state:
    st.session_state.urls = []
if "extracted" not in st.session_state:
    st.session_state.extracted = []


# Schema for Google search results
class GoogleSearchResultSchema(BaseModel):
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Short snippet/preview of the search result content")

class GoogleSearchPageSchema(BaseModel):
    query: str = Field(..., description="Search query used to fetch results")
    results: List[GoogleSearchResultSchema] = Field(..., description="List of search results extracted from Google")


# Cached function to search Google
@st.cache_data
def search_google_for_leads(query: str, firecrawl_api_key: str, num_results: int) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"Authorization": f"Bearer {firecrawl_api_key}", "Content-Type": "application/json"}
    payload = {"query": f"Google Search about {query}", "limit": num_results, "lang": "en", "timeout": 60000}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200 and response.json().get("success"):
        return [result["url"] for result in response.json().get("data", [])]
    return []


# Cached function to extract search info
@st.cache_data
def extract_google_search_info(urls: List[str], firecrawl_api_key: str) -> List[dict]:
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    search_data = []
    urls = urls[:10]
    for url in urls:
        try:
            response = firecrawl_app.extract(
                [url],
                {
                    'prompt': 'Extract title, URL, and snippet from the Google search result page.',
                    'schema': GoogleSearchPageSchema.model_json_schema(),
                }
            )
            if response.get('success') and response.get('status') == 'completed':
                search_data.append({"website_url": url, "search_results": response.get('data', {}).get('results', [])})
        except ValueError as e:
            print(f"Skipping URL due to error: {url} -> {e}")
            continue

    return search_data


# Function to format data
def format_search_data(data: List[Dict]) -> List[Dict]:
    formatted = []
    for entry in data:
        for result in entry.get("search_results", []):
            formatted.append({
                "Website URL": entry.get("website_url", ""),
                "Title": result.get("title", ""),
                "Snippet": result.get("snippet", "")
            })
    return formatted


def extract_clean_google_sheets_link(response_content: str) -> str:
    """Extract and clean the Google Sheets link from response content."""
    match = re.search(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)", response_content)
    if match:
        return f"https://docs.google.com/spreadsheets/d/{match.group(1)}"
    return None


# Search button
if st.button("🔍 Search Google"):
    if firecrawl_api_key:
        with st.spinner("Fetching results..."):
            st.session_state.urls = search_google_for_leads(query, firecrawl_api_key, num_results)
            st.session_state.extracted = extract_google_search_info(st.session_state.urls, firecrawl_api_key)
            st.session_state.formatted_data = format_search_data(st.session_state.extracted)

            if st.session_state.formatted_data:
                st.success(f"✅ Fetched {len(st.session_state.formatted_data)} results!")
                st.dataframe(st.session_state.formatted_data)
            else:
                st.error("❌ No results found!")
    else:
        st.warning("⚠️ Please enter a valid Firecrawl API Key!")


# Initialize Composio Toolset
composio_toolset = ComposioToolSet(api_key="3dms9ab2w9muwy6pa72l3a")

# Get Google Sheets Tool
google_sheets_tool = composio_toolset.get_tools(actions=[Action.GOOGLESHEETS_SHEET_FROM_JSON])[0]

# Create Agent
google_sheets_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[google_sheets_tool],
    show_tool_calls=True,
    instructions="You are an expert at creating and updating Google Sheets. You will be given user information in JSON format, and you need to write it into a new Google Sheet.",
    markdown=True
)

# Upload to Google Sheets
if st.button("📤 Save to Google Sheets"):
    if st.session_state.formatted_data:
        # Create the message with the data directly (no json.dumps)
        message = f"""Create a new Google Sheet with this data and return the Spreadsheet link as well.
        The sheet should have these columns: Website URL, Username, Bio, Post Type, Timestamp, Upvotes, and Links in the same order.
        Here's the data in JSON format:
        {st.session_state.formatted_data}
        """

        with st.spinner("Saving data..."):
            create_sheet_response = google_sheets_agent.run(message, force_tool_calls=True)
            print("------------")
            print(create_sheet_response.content)
            if create_sheet_response and create_sheet_response.content:
                response_content = create_sheet_response.content.strip()

                # Extract and clean the Google Sheets link
                google_sheets_link_clean = extract_clean_google_sheets_link(response_content)
                print(google_sheets_link_clean)
                if google_sheets_link_clean:
                    st.success("✅ Data saved successfully!")
                    st.markdown(f"[📄 Open Google Sheet]({google_sheets_link_clean})", unsafe_allow_html=True)
                else:
                    st.error("❌ Google Sheets link not found in the response.")
            else:
                st.error("❌ Failed to create Google Sheet.")
    else:
        st.warning("⚠️ No data available! Please search first.")
