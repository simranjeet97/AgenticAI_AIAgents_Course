# agent.py
import os
import uuid
import asyncio

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set API key directly (replace this with dotenv in prod)
os.environ["GOOGLE_API_KEY"] = ""  # Replace securely in production

# Configure the root agent
root_agent = Agent(
    name="searching_agent",
    model="gemini-2.0-flash",
    description="Uses Google Search tool to answer questions.",
    instruction="Use the google_search tool to retrieve up-to-date facts if necessary.",
    tools=[google_search],
)

# Initialize session service and runner
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="search_app",
    session_service=session_service,
)

USER_ID = "local_user"
SESSION_ID = "sess_" + uuid.uuid4().hex[:8]

async def get_clean_response(prompt: str) -> str:
    # Create a session
    await session_service.create_session(
        app_name="search_app",
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # Create message content
    content = Content(
        role="user",
        parts=[Part(text=prompt)]
    )

    # Capture final agent response
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.content and event.is_final_response():
            return event.content.parts[0].text.strip()

    return "No response received."

if __name__ == "__main__":
    prompt = "Who is the Prime Minister of India?"
    response = asyncio.run(get_clean_response(prompt))
    print(response)
