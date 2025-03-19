import asyncio
import os
import json
from dataclasses import dataclass
from typing import Optional
import httpx
import dotenv
from pydantic import Field, BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# import logfire
# logfire.configure()

dotenv.load_dotenv()
api_key = os.getenv("CBORG_API_KEY")


@dataclass
class ApiDeps:
    """Dependencies container for the API agent."""
    client: httpx.AsyncClient


ai_model = OpenAIModel(
    model_name="openai/gpt-4o",
    provider=OpenAIProvider(
        base_url="https://api.cborg.lbl.gov/v1",
        api_key=api_key,
    )
)

wikipedia_api_agent = Agent(
    ai_model,
    system_prompt=(
        "You are a helpful assistant that can give any answers to Animals that are on Wikipedia. Do not use your own knowledge."
    ),
    deps_type=ApiDeps,
    instrument=True
)


@wikipedia_api_agent.tool()
async def get_animal_info(ctx: RunContext[ApiDeps], animal_name: str) -> str:
    """
    Get information about an animal using the Wikipedia API.

    This function searches for the specified animal on Wikipedia and returns
    a summary of information about it. Use this function whenever you need
    to provide factual information about animals, including:
    - Animal descriptions
    - Habitats
    - Diets
    - Behavior
    - Conservation status

    Args:
        ctx: The run context
        animal_name: The name of the animal to look up (e.g., "tiger", "blue whale", "emperor penguin")

    Returns:
        A string containing information about the animal from Wikipedia
    """
    try:
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{animal_name} animal",
            "format": "json"
        }
        search_response = await ctx.deps.client.get("https://en.wikipedia.org/w/api.php", params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()

        if not search_data["query"]["search"]:
            return f"No information found for {animal_name}."

        page_title = search_data["query"]["search"][0]["title"]

        # summary content
        content_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": page_title,
            "format": "json"
        }
        content_response = await ctx.deps.client.get("https://en.wikipedia.org/w/api.php", params=content_params)
        content_response.raise_for_status()
        content_data = content_response.json()

        pages = content_data["query"]["pages"]
        page_id = next(iter(pages))
        extract = pages[page_id].get("extract", "")

        if not extract:
            return f"Found page {page_title} but couldn't extract any information."

        # Return a concise version
        return f"Information about {page_title} from Wikipedia:\n\n{extract[:800]}..."

    except Exception as e:
        return f"Error retrieving information about {animal_name}: {str(e)}"


def run_query(query: str) -> None:
    """
    Run a query against the wikipedia_api_agent synchronously.

    This function creates a synchronous HTTP client, injects the necessary
    dependencies, and prints the agent's response.

    Args:
        query (str): The question to be processed by the agent.
    """
    logger.info("Running query: %s", query)
    with httpx.Client() as client:
        deps = ApiDeps(client=client)
        result = wikipedia_api_agent.run_sync(query, deps=deps)
        print(result)