import asyncio
import logging
import os
import random
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Annotated, Any

from agent_framework import Agent, BaseHistoryProvider, Message, tool
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import Field
from rich import print
from rich.logging import RichHandler

# Configurar logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configurar cliente de OpenAI según el entorno
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


class SQLiteHistoryProvider(BaseHistoryProvider):
    """Un proveedor de historial personalizado respaldado por SQLite.

    Implementa BaseHistoryProvider para persistir mensajes de chat
    en una base SQLite local: es útil cuando quieres persistencia
    basada en archivos sin un servicio externo como Redis.
    """

    def __init__(self, db_path: str):
        super().__init__("sqlite-history")
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    async def get_messages(self, session_id: str | None, **kwargs: Any) -> list[Message]:
        """Recupera todos los mensajes de esta sesión desde SQLite."""
        if session_id is None:
            return []
        cursor = self._conn.execute(
            "SELECT message_json FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [Message.from_json(row[0]) for row in cursor.fetchall()]

    async def save_messages(self, session_id: str | None, messages: Sequence[Message], **kwargs: Any) -> None:
        """Guarda mensajes en la base de datos SQLite."""
        if session_id is None:
            return
        self._conn.executemany(
            "INSERT INTO messages (session_id, message_json) VALUES (?, ?)",
            [(session_id, message.to_json()) for message in messages],
        )
        self._conn.commit()

    def close(self) -> None:
        """Cierra la conexión a SQLite."""
        self._conn.close()


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Devuelve datos del clima para una ciudad."""
    logger.info(f"Obteniendo el clima para {city}")
    conditions = ["soleado", "nublado", "lluvioso", "tormentoso"]
    return f"El clima en {city} está {conditions[random.randint(0, 3)]} con una máxima de {random.randint(10, 30)}°C."


async def main() -> None:
    """Demuestra una sesión con SQLite que persiste el historial en un archivo local."""
    db_path = "chat_history.sqlite3"
    session_id = str(uuid.uuid4())

    # Fase 1: Iniciar una conversación con un proveedor de historial en SQLite
    print("\n[bold]=== Sesión persistente en SQLite ===[/bold]")
    print("[dim]--- Fase 1: Iniciando conversación ---[/dim]")

    sqlite_provider = SQLiteHistoryProvider(db_path=db_path)

    agent = Agent(
        client=client,
        instructions="Eres un agente de clima útil.",
        tools=[get_weather],
        context_providers=[sqlite_provider],
    )

    session = agent.create_session(session_id=session_id)

    print("[blue]Usuario:[/blue] ¿Cómo está el clima en Tokio?")
    response = await agent.run("¿Cómo está el clima en Tokio?", session=session)
    print(f"[green]Agente:[/green] {response.text}")

    print("\n[blue]Usuario:[/blue] ¿Y París?")
    response = await agent.run("¿Y París?", session=session)
    print(f"[green]Agente:[/green] {response.text}")

    messages = await sqlite_provider.get_messages(session_id)
    print(f"[dim]Mensajes guardados en SQLite: {len(messages)}[/dim]")
    sqlite_provider.close()

    # Fase 2: Simular un reinicio de la app — reconectar al mismo session_id en SQLite
    print("\n[dim]--- Fase 2: Reanudando después del 'reinicio' ---[/dim]")
    sqlite_provider2 = SQLiteHistoryProvider(db_path=db_path)

    agent2 = Agent(
        client=client,
        instructions="Eres un agente de clima útil.",
        tools=[get_weather],
        context_providers=[sqlite_provider2],
    )

    session2 = agent2.create_session(session_id=session_id)

    print("[blue]Usuario:[/blue] ¿Cuál de las ciudades por las que pregunté tuvo mejor clima?")
    response = await agent2.run("¿Cuál de las ciudades por las que pregunté tuvo mejor clima?", session=session2)
    print(f"[green]Agente:[/green] {response.text}")
    print("[dim]Nota: El agente recordó la conversación de la Fase 1 gracias a la persistencia en SQLite.[/dim]")

    sqlite_provider2.close()

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
