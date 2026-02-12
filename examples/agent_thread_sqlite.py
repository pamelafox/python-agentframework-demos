import asyncio
import logging
import os
import random
import sqlite3
import uuid
from collections.abc import MutableMapping, Sequence
from typing import Annotated, Any

from agent_framework import AgentThread, ChatAgent, ChatMessage, ChatMessageStoreProtocol
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
    client = OpenAIChatClient(
        api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    )


class SQLiteChatMessageStore(ChatMessageStoreProtocol):
    """A custom ChatMessageStore backed by SQLite.

    Implements the ChatMessageStoreProtocol to persist chat messages
    in a local SQLite database — useful when you want file-based
    persistence without an external service like Redis.
    """

    def __init__(self, db_path: str, thread_id: str | None = None):
        self.db_path = db_path
        self.thread_id = thread_id or str(uuid.uuid4())
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Add messages to the SQLite database."""
        self._conn.executemany(
            "INSERT INTO messages (thread_id, message_json) VALUES (?, ?)",
            [(self.thread_id, message.to_json()) for message in messages],
        )
        self._conn.commit()

    async def list_messages(self) -> list[ChatMessage]:
        """Retrieve all messages for this thread from SQLite."""
        cursor = self._conn.execute(
            "SELECT message_json FROM messages WHERE thread_id = ? ORDER BY id",
            (self.thread_id,),
        )
        return [ChatMessage.from_json(row[0]) for row in cursor.fetchall()]

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the store state for persistence."""
        return {"db_path": self.db_path, "thread_id": self.thread_id}

    @classmethod
    async def deserialize(
        cls, serialized_store_state: MutableMapping[str, Any], **kwargs: Any
    ) -> "SQLiteChatMessageStore":
        """Reconstruct a store from serialized state."""
        return cls(
            db_path=serialized_store_state["db_path"],
            thread_id=serialized_store_state["thread_id"],
        )

    async def update_from_state(self, serialized_store_state: MutableMapping[str, Any], **kwargs: Any) -> None:
        """Update store from serialized state."""
        if serialized_store_state:
            self.db_path = serialized_store_state["db_path"]
            self.thread_id = serialized_store_state["thread_id"]

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info(f"Getting weather for {city}")
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {conditions[random.randint(0, 3)]} with a high of {random.randint(10, 30)}°C."


async def main() -> None:
    """Demonstrate a SQLite-backed thread that persists conversation history to a local file."""
    db_path = "chat_history.sqlite3"
    thread_id = str(uuid.uuid4())

    # Phase 1: Start a conversation with a SQLite-backed thread
    print("\n[bold]=== Persistent SQLite Thread ===[/bold]")
    print("[dim]--- Phase 1: Starting conversation ---[/dim]")

    store = SQLiteChatMessageStore(db_path=db_path, thread_id=thread_id)
    thread = AgentThread(message_store=store)

    agent = ChatAgent(
        chat_client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    print("[blue]User:[/blue] What's the weather like in Tokyo?")
    response = await agent.run("What's the weather like in Tokyo?", thread=thread)
    print(f"[green]Agent:[/green] {response.text}")

    print("\n[blue]User:[/blue] How about Paris?")
    response = await agent.run("How about Paris?", thread=thread)
    print(f"[green]Agent:[/green] {response.text}")

    messages = await store.list_messages()
    print(f"[dim]Messages stored in SQLite: {len(messages)}[/dim]")
    store.close()

    # Phase 2: Simulate an application restart — reconnect to the same thread ID in SQLite
    print("\n[dim]--- Phase 2: Resuming after 'restart' ---[/dim]")
    store2 = SQLiteChatMessageStore(db_path=db_path, thread_id=thread_id)
    thread2 = AgentThread(message_store=store2)

    agent2 = ChatAgent(
        chat_client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    print("[blue]User:[/blue] Which of the cities I asked about had better weather?")
    response = await agent2.run("Which of the cities I asked about had better weather?", thread=thread2)
    print(f"[green]Agent:[/green] {response.text}")
    print("[dim]Note: The agent remembered the conversation from Phase 1 via SQLite persistence.[/dim]")

    store2.close()

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
