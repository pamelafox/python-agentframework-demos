import asyncio
import logging
import os
import random
from typing import Annotated

from agent_framework import AgentThread, ChatAgent, ChatMessageStore
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import Field
from rich import print
from rich.logging import RichHandler

# Setup logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure OpenAI client based on environment
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

async_credential = None
if API_HOST == "azure":
    async_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(async_credential, "https://cognitiveservices.azure.com/.default")
    client = OpenAIChatClient(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
        api_key=token_provider,
        model_id=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )
elif API_HOST == "github":
    client = OpenAIChatClient(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-5-mini"),
    )
else:
    client = OpenAIChatClient(api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))


def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info(f"Getting weather for {city}")
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {conditions[random.randint(0, 3)]} with a high of {random.randint(10, 30)}°C."


agent = ChatAgent(
    chat_client=client,
    instructions="You are a helpful weather agent.",
    tools=[get_weather],
)


async def example_without_thread() -> None:
    """Without a thread, each call is independent — the agent has no memory of prior messages."""
    print("\n[bold]=== Without Thread (No Memory) ===[/bold]")

    response = await agent.run("What's the weather like in Seattle?")
    print(f"[blue]User:[/blue] What's the weather like in Seattle?")
    print(f"[green]Agent:[/green] {response.text}")

    response = await agent.run("What was the last city I asked about?")
    print(f"\n[blue]User:[/blue] What was the last city I asked about?")
    print(f"[green]Agent:[/green] {response.text}")
    print("[dim]Note: Each call creates a separate thread, so the agent doesn't remember previous context.[/dim]")


async def example_with_thread() -> None:
    """With a thread, the agent maintains context across multiple messages."""
    print("\n[bold]=== With Thread (Persistent Memory) ===[/bold]")

    thread = agent.get_new_thread()

    print(f"[blue]User:[/blue] What's the weather like in Tokyo?")
    response = await agent.run("What's the weather like in Tokyo?", thread=thread)
    print(f"[green]Agent:[/green] {response.text}")

    print(f"\n[blue]User:[/blue] How about London?")
    response = await agent.run("How about London?", thread=thread)
    print(f"[green]Agent:[/green] {response.text}")

    print(f"\n[blue]User:[/blue] Which of those cities has better weather?")
    response = await agent.run("Which of those cities has better weather?", thread=thread)
    print(f"[green]Agent:[/green] {response.text}")
    print("[dim]Note: The agent remembers context from previous messages in the same thread.[/dim]")


async def example_thread_across_agents() -> None:
    """A thread's message history can be carried over to a new agent instance."""
    print("\n[bold]=== Thread Across Agent Instances ===[/bold]")

    thread = agent.get_new_thread()

    print(f"[blue]User:[/blue] What's the weather in Paris?")
    response = await agent.run("What's the weather in Paris?", thread=thread)
    print(f"[green]Agent 1:[/green] {response.text}")

    if thread.message_store:
        messages = await thread.message_store.list_messages()
        print(f"[dim]Thread contains {len(messages or [])} messages[/dim]")

    # Create a second agent and continue with the same thread
    agent2 = ChatAgent(
        chat_client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    print(f"\n[blue]User:[/blue] What was the last city I asked about?")
    response = await agent2.run("What was the last city I asked about?", thread=thread)
    print(f"[green]Agent 2:[/green] {response.text}")
    print("[dim]Note: The second agent continues the conversation using the thread's message history.[/dim]")

    # You can also create a new thread from existing messages
    messages = await thread.message_store.list_messages() if thread.message_store else []
    new_thread = AgentThread(message_store=ChatMessageStore(messages))

    print(f"\n[blue]User:[/blue] How does Paris weather compare to London?")
    response = await agent2.run("How does Paris weather compare to London?", thread=new_thread)
    print(f"[green]Agent 2 (new thread):[/green] {response.text}")
    print("[dim]Note: A new thread was created from the old thread's messages — conversation continues.[/dim]")


async def main() -> None:
    """Run all thread examples to demonstrate different persistence patterns."""
    await example_without_thread()
    await example_with_thread()
    await example_thread_across_agents()

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
