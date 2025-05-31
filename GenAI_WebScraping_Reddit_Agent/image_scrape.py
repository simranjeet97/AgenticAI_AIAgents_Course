import os
import requests
import re
import streamlit as st
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
from bs4 import BeautifulSoup
from composio_agno import Action, ComposioToolSet
from agno.agent import Agent
from agno.models.google import Gemini

# Set Google Gemini API key
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Set Firecrawl API key
firecrawl_api_key = "fc-67de2fa4ad80497dbf196bdef571ecc9"

# Initialize Firecrawl App
firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)

# Schema for Amazon product data
class AmazonProductSchema(BaseModel):
    product_title: Optional[str] = Field(None, description="Title of the product")
    price: Optional[str] = Field(None, description="Price of the product")
    image_urls: List[str] = Field(default=[], description="List of image URLs of the product")
    description: Optional[str] = Field(None, description="Description of the product")

# Function to fetch Amazon product links based on search term
def fetch_product_links(search_term: str) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:110.0) Gecko/20100101 Firefox/110.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    search_url = f"https://www.amazon.com/s?k={search_term.replace(' ', '+')}"
    response = session.get(search_url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        product_links = []
        
        for link in soup.find_all("a", href=True):
            if "/dp/" in link["href"]:  # Look for ASIN pattern
                asin = re.search(r"/dp/([A-Z0-9]{10})", link["href"])  # Extract ASIN
                if asin:
                    asin_code = asin.group(1)
                    product_links.append(f"https://www.amazon.com/dp/{asin_code}")
        
        return product_links
    else:
        st.error("❌ Failed to fetch Amazon search results.")
        return []

# Function to extract product data from a URL
def extract_product_data(urls: List[str]) -> List[Dict]:
    search_data = []
    urls = urls[:10]
    for url in urls:
        response = firecrawl_app.extract(
            urls=[url],
            params={
                "prompt": "Extract the product title, price, image URLs, and description from this Amazon product page.",
                "schema": AmazonProductSchema.model_json_schema()
            }
        )

        if response.get("success") and response.get("status") == "completed":
            search_data.append({
                "website_url": url,
                "search_results": response.get("data", {})
            })
        else:
            st.warning(f"❌ Extraction failed for {url}: {response}")
    
    return search_data

# Function to format extracted product data for Google Sheets
def format_search_data(data: List[Dict]) -> List[Dict]:
    formatted = []
    for entry in data:
        search_results = entry.get("search_results", {})
        formatted.append({
            "Website URL": entry.get("website_url", ""),
            "Title": search_results.get("product_title", ""),
            "Price": search_results.get("price", ""),
            "Image URLs": ", ".join(search_results.get("image_urls", [])),  # Convert list to comma-separated string
            "Description": search_results.get("description", "")
        })
    return formatted

# Function to create a Google Sheet with the extracted and formatted data
def create_google_sheet(formatted_data: List[Dict]) -> str:
    # Initialize Composio Toolset
    composio_toolset = ComposioToolSet(api_key="3dms9ab2w9muwy6pa72l3a")

    # Get Google Sheets Tool
    google_sheets_tool = composio_toolset.get_tools(actions=[Action.GOOGLESHEETS_SHEET_FROM_JSON])[0]

    # Create Agent
    google_sheets_agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        tools=[google_sheets_tool],
        show_tool_calls=True,
        instructions="You are an expert at creating and updating Google Sheets. You will be given user information in JSON format, and you need to write it into a new Google Sheet.",
        markdown=True
    )

    # Create the message with the data directly (no json.dumps)
    message = f"""
    Create a new Google Sheet with the following data and return the Google Sheets link.
    The sheet should have these columns in the exact order: Website URL, Title, Price, Image URLs, and Description.
    Here's the data in JSON format:
    {formatted_data}
    """

    # Call the agent with the message
    create_sheet_response = google_sheets_agent.run(
        message,
        force_tool_calls=True
    )

    # Extract Google Sheet link from the response
    response_content = create_sheet_response.content.strip()
    match = re.search(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)", response_content)
    if match:
        sheet_link = f"https://docs.google.com/spreadsheets/d/{match.group(1)}"
        return sheet_link
    else:
        return "❌ Failed to create Google Sheet."

# Streamlit UI layout with stylish elements
def display_ui():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f1f1f1;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>input {
        border-radius: 8px;
        border: 2px solid #4CAF50;
        padding: 10px;
        width: 100%;
    }
    .stTextInput>input:focus {
        border: 2px solid #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page Title
    st.title("Amazon Product Scraper and Google Sheets Exporter")
    st.markdown("""
    **Welcome to the Amazon Product Scraper!**
    This tool allows you to scrape product details from Amazon and export them to Google Sheets.
    """)

    # User input
    search_term = st.text_input("Enter product search term:", "gaming pcs", max_chars=100)

    # Scrape Button
    if st.button("🛒 Scrape Products"):
        with st.spinner("Fetching product links from Amazon..."):
            product_links = fetch_product_links(search_term)
            st.write(f"Found {len(product_links)} product links.")

            if product_links:
                with st.spinner("Extracting product data..."):
                    search_data = extract_product_data(product_links)

                if search_data:
                    formatted_data = format_search_data(search_data)

                    st.write("### Extracted Product Data:")
                    st.write(formatted_data)

                    with st.spinner("Creating Google Sheet..."):
                        sheet_link = create_google_sheet(formatted_data)

                    if "https://docs.google.com/spreadsheets/d/" in sheet_link:
                        st.success(f"✅ Google Sheet created successfully: [Open Sheet]({sheet_link})")
                    else:
                        st.error("❌ Failed to create Google Sheet.")
                else:
                    st.warning("No product data was extracted.")
            else:
                st.warning("No product links found.")

# Run the Streamlit app
if __name__ == "__main__":
    display_ui()