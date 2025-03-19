import os
from geopy.geocoders import Nominatim
from meteostat import Point, Daily
from pydantic_ai import Agent
from dateutil import parser
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Any

# Load CBORG API key from environment variable
api_key = os.getenv("CBORG_API_KEY")

# Ensure the API key is set
if not api_key:
    raise ValueError("CBORG_API_KEY environment variable is not set.")

geo = Nominatim(user_agent="EGSB Hackathon AI Agent toy")

# Configure the AI model with CBORG API endpoint
ai_model = OpenAIModel(
    model_name="anthropic/claude-sonnet",
    provider=OpenAIProvider(
        base_url="https://api.cborg.lbl.gov",
        api_key=api_key,
    )
)

# Initialize the Agent with a system prompt
geo_agent = Agent(
    ai_model,
    system_prompt='You are an awesome geography teacher.',
)

# Register a tool to get elevation for given latitude and longitude
@geo_agent.tool_plain
def get_loc(location_string) -> (float, float, float):
    """
    Get the lat / long for a place gien an address string or other location
    string.
    location_string - the location to query like an address a city / state / country etc.

    Returns a tuple of the latitude abd,longitude of the location.
    """
    loc = geo.geocode(location_string)
    print(loc)
    print()
    return loc.latitude, loc.longitude

@geo_agent.tool_plain
def get_weather(location_string: str, start_date: str, end_date: str) -> dict[str, Any]:
    """
    Get information about the weather at a particular location over a particlar time period.

    location_string - the location to query like an address a city / state / country etc.
    start_date - the start of the period of inteest as a string.
    end_date - the end of the period of inteest as a string.

    Returns a dictionary of weather information for the location.
    """
    pt = Point(*get_loc(location_string))
    st = parser.parse(start_date)
    end = parser.parse(end_date)
    print(pt, start_date, st, end_date, end)
    ret = Daily(pt, st, end).fetch()
    print(ret)
    d = ret.to_dict()
    print(d)
    return d

# Use the agent to query elevation
result = geo_agent.run_sync(
    """
    Tell me about the weather in the city of kalamazoo? over 7 days from Feburary 14 ,2024.
    Summarize the general trends and how happy you think people would be about the weather.
    """
)
print(result)
