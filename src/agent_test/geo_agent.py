import os
from typing import List

from nmdc_geoloc_tools import elevation
from pydantic_ai import Agent, ModelRetry, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_test.maptools import get_static_map

api_key = os.getenv("CBORG_API_KEY")
ai_model = OpenAIModel(
    "openai/gpt-4o",
    provider=OpenAIProvider(
        base_url="https://api.cborg.lbl.gov",
        api_key=api_key),
)


geo_agent = Agent(
    ai_model,
    system_prompt="""You are an awesome geography teacher.
    You can use the following tools to help you answer questions:
    
    `get_elev`: Get the elevation of a location.
    `fetch_map_image_and_interpret`: Fetch an image of a location and describe it.
    
    Note that when interpreting images, you might want to try different zoom levels
    and switching between roadmap and satellite to get an overall sense of what is there.
    """
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
    print(f"Looking up elevation for lat={lat}, lon={lon}")
    return elevation((lat, lon))

map_reader_agent = Agent(
    ai_model,
    system_prompt='Your job is to interpret images of maps.',
)

@geo_agent.tool_plain
async def fetch_map_image_and_interpret(lat: float, lon: float, zoom=18, maptype="satellite") -> List[str]:
    """
    Fetch an image of a location and describe it.

    You may want to try running this different times at different zoom levels to help interpret.

    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        zoom: Zoom level for the map (18 is good for zoomed in, 13 for further out)
        maptype: Type of map (e.g., "satellite", "roadmap")

    Returns:
        list: list of descriptions
    """
    print(f"Fetching map image for lat={lat}, lon={lon}")
    img_bytes = get_static_map(lat, lon, zoom=zoom, maptype=maptype)
    img = BinaryContent(data=img_bytes, media_type='image/png')
    if not img:
        raise ModelRetry("Could not find image for structure")

    r = await map_reader_agent.run(
        [f"""list all of the features you see in this {maptype} image.
            Only give me actual features, I don't care that I am looking at a google map.
            In particular, environmental features, or kinds of buildings. Give your best guess,
            but tell me if you are not sure. If it's zoomed in too far or out too far, say this
            in your response.
            """, img],
        )
    print("INTERPRETATION", r.data)
    return r.data

result = geo_agent.run_sync('What features do you see at 35.97583846 and long=-84.2743123')
#result = geo_agent.run_sync('How high is the location on earth with lat=35.97583846 and long=-84.2743123')
#print(result.all_messages_json())  # type: ignore
