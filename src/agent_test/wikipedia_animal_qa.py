import os
import json
import asyncio
from dataclasses import dataclass
from typing import List, Any

import httpx
import dotenv

# Try importing gradio and error out if not installed.
try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `pip install gradio`. You must use python>=3.10.'
    ) from e

# Import pydantic_ai modules for the agent and streaming responses.
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

# Load environment variables from a .env file (ensure CBORG_API_KEY is set)
dotenv.load_dotenv()
api_key = os.getenv("CBORG_API_KEY")
if not api_key:
    raise ValueError("CBORG_API_KEY environment variable is not set.")


# Define a dependencies container for dependency injection.
@dataclass
class ApiDeps:
    client: httpx.AsyncClient


# Configure the AI model using your CBORG API endpoint.
ai_model = OpenAIModel(
    model_name="openai/gpt-4o",
    provider=OpenAIProvider(
        base_url="https://api.cborg.lbl.gov/v1",
        api_key=api_key,
    )
)

# Initialize the Wikipedia agent with a system prompt and dependency type.
wikipedia_api_agent = Agent(
    ai_model,
    system_prompt=(
        "You are a helpful assistant that provides animal information exclusively from Wikipedia. "
        "Do not use your own pre-existing knowledge."
    ),
    deps_type=ApiDeps,
    instrument=True
)


# Register a tool to get animal information from Wikipedia.
@wikipedia_api_agent.tool()
async def get_animal_info(ctx: RunContext[ApiDeps], animal_name: str) -> str:
    """
    Get information about an animal using the Wikipedia API.

    This function searches for the specified animal on Wikipedia and returns a summary
    of the information found.
    """
    try:
        # Search for the animal's page.
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

        # Use the first result.
        page_title = search_data["query"]["search"][0]["title"]

        # Retrieve the introductory extract for the page.
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

        # Return a concise version.
        return f"Information about {page_title} from Wikipedia:\n\n{extract[:800]}..."
    except Exception as e:
        return f"Error retrieving information about {animal_name}: {str(e)}"


# --- Gradio Integration Section ---

# Create an Async HTTP client and instantiate dependency container.
client = httpx.AsyncClient()
deps = ApiDeps(client=client)

# Define a mapping for tool display names (if the agent calls any tool).
TOOL_TO_DISPLAY_NAME = {
    "get_animal_info": "Wikipedia API"
}


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    """
    Asynchronously stream responses from the Wikipedia agent.

    The user's prompt is added to the chat history, and then the agent's
    streaming response (including any tool calls) is yielded.
    """
    chatbot.append({"role": "user", "content": prompt})
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip()

    async with wikipedia_api_agent.run_stream(prompt, deps=deps, message_history=past_messages) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    # Check if call.args is a string or an object with attributes
                    if isinstance(call.args, str):
                        call_args = call.args
                    elif hasattr(call.args, "args_json"):
                        call_args = call.args.args_json
                    elif hasattr(call.args, "args_dict"):
                        call_args = json.dumps(call.args.args_dict)
                    else:
                        call_args = str(call.args)

                    metadata = {"title": f"🛠️ Using {TOOL_TO_DISPLAY_NAME.get(call.tool_name, call.tool_name)}"}
                    if call.tool_call_id is not None:
                        metadata["id"] = call.tool_call_id  # Assign directly, not as a set

                    gr_message = {
                        "role": "assistant",
                        "content": "Parameters: " + call_args,
                        "metadata": metadata,
                    }
                    chatbot.append(gr_message)
                if isinstance(call, ToolReturnPart):
                    for gr_message in chatbot:
                        if gr_message.get("metadata", {}).get("id", "") == call.tool_call_id:
                            gr_message["content"] += f"\nOutput: {json.dumps(call.content)}"
            yield gr.skip(), chatbot, gr.skip()

        chatbot.append({"role": "assistant", "content": ""})
        async for message in result.stream_text():
            chatbot[-1]["content"] = message
            yield gr.skip(), chatbot, gr.skip()

        past_messages = result.all_messages()
        yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    """
    Handle retry requests by the user.
    """
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    """
    Handle undo events.
    """
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    """
    Select data from an example message.
    """
    return message.value["text"]


# Build the Gradio UI.
with gr.Blocks() as demo:
    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
    <img src="https://ai.pydantic.dev/img/logo-white.svg" style="max-width: 200px; height: auto">
    <div>
        <h1 style="margin: 0 0 1rem 0">Wikipedia Animal Info Assistant</h1>
        <h3 style="margin: 0 0 0.5rem 0">
            Ask me about animals, and I'll fetch information from Wikipedia.
        </h3>
    </div>
</div>
"""
    )

    past_messages = gr.State([])
    chatbot = gr.Chatbot(
        label="Wikipedia Assistant",
        type="messages",
        avatar_images=(None, "https://ai.pydantic.dev/img/logo-white.svg"),
        examples=[
            {"text": "Tell me about the tiger."},
            {"text": "What information do you have on the blue whale?"},
        ],
    )

    with gr.Row():
        prompt = gr.Textbox(
            lines=1,
            show_label=False,
            placeholder='Ask about an animal, e.g., "Tell me about the elephant."',
        )

    # Set up the submission to stream responses from the agent.
    generation = prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages],
        outputs=[prompt, chatbot, past_messages],
    )

    # Enable selecting, retrying, and undoing messages.
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages])
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])

if __name__ == "__main__":
    demo.launch()