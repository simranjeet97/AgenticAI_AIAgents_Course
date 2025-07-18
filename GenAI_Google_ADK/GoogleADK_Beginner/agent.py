import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
import os

os.environ["GOOGLE_API_KEY"] = "" # <--- REPLACE

def get_weather(city: str) -> dict:
    """Get the current weather in a city."""
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": "The weather in New York is sunny with 25°C."
        }
    return {
        "status": "error",
        "error_message": f"Weather for '{city}' unavailable."
    }
def get_current_time(city: str) -> dict:
    """Get the current time in a city."""
    city_timezones = {
        "new york": "America/New_York",
        "london": "Europe/London",
        "tokyo": "Asia/Tokyo",
        "paris": "Europe/Paris"
    }
    if city.lower() in city_timezones:
        try:
            tz = ZoneInfo(city_timezones[city.lower()])
            now = datetime.datetime.now(tz)
            return {
                "status": "success",
                "report": f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            }
        except Exception:
            pass
    return {
        "status": "error",
        "error_message": f"Time information for '{city}' unavailable."
    }
# Define the agent with the name "root_agent" (required by ADK)
root_agent = Agent(
    name="weather_time_agent",
    model="gemini-1.5-flash",  # Use your preferred Gemini model
    description="Agent that provides weather and time information for cities.",
    instruction="You help users with time and weather information for various cities.",
    tools=[get_weather, get_current_time],
)

if __name__ == "__main__":
    # Agent code here
    print("Agent initialized.")