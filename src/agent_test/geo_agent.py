import os

from nmdc_geoloc_tools import elevation
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

api_key = os.getenv("CBORG_API_KEY")
ai_model = OpenAIModel(
    "openai/gpt-4o",
    provider=OpenAIProvider(
        base_url="https://api.cborg.lbl.gov",
        api_key=api_key),
)


geo_agent = Agent(
    ai_model,
    system_prompt='You are an awesome geography teacher.',
)

@geo_agent.tool_plain
def get_elev(
    lat: float, lon: float,
) -> float:
    """
    Get the elevation of a location.

    :param lat: latitude
    :param lon: longitude
    :return: elevation in m
    """
    return elevation((lat, lon))

result = geo_agent.run_sync('How high is the location on earth with lat=27.9881 and long=86.9250')
print(result.data)
