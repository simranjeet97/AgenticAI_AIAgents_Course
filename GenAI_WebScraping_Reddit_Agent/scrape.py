import os
import streamlit as st
import requests
import json
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from firecrawl import FirecrawlApp
from composio_agno import Action, ComposioToolSet
from typing import List
from pydantic import BaseModel, Field
from agno.models.google import Gemini

# Set Google Gemini API key
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Define schemas for extracted Quora data
class QuoraUserInteractionSchema(BaseModel):
    username: str = Field(description="The username of the Quora user")
    bio: str = Field(description="The bio or description of the user")
    post_type: str = Field(description="Type of post, either 'question' or 'answer'")
    timestamp: str = Field(description="When the post was made")
    upvotes: int = Field(default=0, description="Number of upvotes received")
    links: List[str] = Field(default_factory=list, description="Links in the post")

class QuoraPageSchema(BaseModel):
    interactions: List[QuoraUserInteractionSchema] = Field(description="List of all user interactions on the page")

# Search for Quora URLs matching a business need
def search_quora_urls(query: str, firecrawl_api_key: str, num_links: int) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"Authorization": f"Bearer {firecrawl_api_key}", "Content-Type": "application/json"}
    payload = {"query": f"Quora discussions about {query}", "limit": num_links, "lang": "en", "timeout": 60000}
    
    response = requests.post(url, json=payload, headers=headers)
    return [result["url"] for result in response.json().get("data", [])] if response.status_code == 200 else []

# Extract user information from Quora URLs
def extract_quora_user_info(urls: List[str], firecrawl_api_key: str) -> List[dict]:
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    user_data = []
    
    for url in urls:
        response = firecrawl_app.extract(
            [url],
            {
                'prompt': 'Extract usernames, bios, post type, timestamps, upvotes, and links from Quora discussions.',
                'schema': QuoraPageSchema.model_json_schema(),
            }
        )
        if response.get('success') and response.get('status') == 'completed':
            user_data.append({"website_url": url, "user_info": response.get('data', {}).get('interactions', [])})
    return user_data

# Flatten extracted Quora data into JSON
def format_user_data(data: List[dict]) -> List[dict]:
    formatted = []
    for entry in data:
        for interaction in entry["user_info"]:
            formatted.append({
                "Website URL": entry["website_url"],
                "Username": interaction.get("username", ""),
                "Bio": interaction.get("bio", ""),
                "Post Type": interaction.get("post_type", ""),
                "Timestamp": interaction.get("timestamp", ""),
                "Upvotes": interaction.get("upvotes", 0),
                "Links": ", ".join(interaction.get("links", [])),
            })
    return formatted

# Composio integration: Write data to Google Sheets
def write_to_google_sheets(data: List[dict], composio_api_key: str) -> str:
    # Get Google Sheets Tool
    composio_toolset = ComposioToolSet(api_key=composio_api_key)
    google_sheets_tool = composio_toolset.get_tools(actions=[Action.GOOGLESHEETS_SHEET_FROM_JSON])[0]

    # Create Agent
    google_sheets_agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[google_sheets_tool],
        show_tool_calls=True,
        instructions="You are an expert at creating and updating Google Sheets. You will be given user information in JSON format, and you need to write it into a new Google Sheet.",
        markdown=True
    )

    # Create the message with the data directly (no json.dumps)
    message = f"""Create a new Google Sheet with this data and return the Spreadsheet link as well.
    The sheet should have these columns: Website URL, Username, Bio, Post Type, Timestamp, Upvotes, and Links in the same order.
    Here's the data in JSON format:
    {data}
    """


    # Call the agent with structured data
    response = google_sheets_agent.run(
        message,
        force_tool_calls=True
    )
    
    if "https://docs.google.com/spreadsheets/d/" in response.content:
            google_sheets_link = response.content.split("https://docs.google.com/spreadsheets/d/")[1].split(" ")[0]
            return f"https://docs.google.com/spreadsheets/d/{google_sheets_link}"
    return response.content if "https://docs.google.com/spreadsheets/d/" in response.content else None

# Streamlit UI
def main():
    st.title("📌 AI Quora Lead Generator")
    
    with st.sidebar:
        st.header("API Keys")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")
        composio_api_key = st.text_input("Composio API Key", type="password")
        num_links = st.number_input("Number of Quora links to fetch", min_value=1, max_value=10, value=3)
    
    user_query = st.text_area("Enter a product/service to find leads:", placeholder="e.g., AI marketing automation tool")
    
    if st.button("Find Leads"):
        if not all([firecrawl_api_key, composio_api_key, user_query]):
            st.error("Please enter all required details!")
            return
        
        with st.spinner("Finding relevant Quora discussions..."):
            urls = search_quora_urls(user_query, firecrawl_api_key, num_links)
        
        if urls:
            st.subheader("Quora Links:")
            for url in urls:
                st.write(url)
            
            with st.spinner("Extracting user information..."):
                user_data = extract_quora_user_info(urls, firecrawl_api_key)
            
            with st.spinner("Formatting data..."):
                formatted_data = format_user_data(user_data)
            
            with st.spinner("Writing data to Google Sheets..."):
                sheet_link = write_to_google_sheets(formatted_data, composio_api_key)
            
            if sheet_link:
                st.success("Data successfully written to Google Sheets!")
                st.success(f"✅ Leads saved to [Google Sheets]({sheet_link.replace('`','')})")
            else:
                st.error("Failed to create Google Sheet.")
        else:
            st.warning("No relevant Quora links found.")

if __name__ == "__main__":
    main()