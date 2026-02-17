"""Compactación de contexto mediante middleware de resumen.

Cuando una conversación crece, los mensajes acumulados pueden exceder la
ventana de contexto del modelo o volverse costosos. Este middleware
monitorea el uso acumulado de tokens y, cuando se cruza un umbral, le pide
al LLM resumir la conversación hasta ese momento. El resumen reemplaza los
mensajes anteriores y libera espacio para futuros turnos.

Diagrama:

 agent.run("mensaje del usuario")
 │
 ▼
 ┌──────────────────────────────────────────────────┐
 │        SummarizationMiddleware (nivel agente)     │
 │                                                  │
 │  1. Revisar uso acumulado de tokens              │
 │  2. Si pasa el umbral → resumir mensajes previos │
 │     con el LLM y reemplazarlos por el resumen    │
 │  3. call_next() → ejecución normal del agente    │
 │  4. Registrar tokens nuevos de la respuesta      │
 └──────────────────────────────────────────────────┘
 │
 ▼
 respuesta

Esto usa middleware a nivel de agente porque el resumen debe ocurrir
*antes* del procesamiento normal del agente (herramientas, chat, etc.) y
necesita acceso al historial completo de mensajes.
"""

import asyncio
import logging
import os
import random
import sys
from collections.abc import Awaitable, Callable
from typing import Annotated

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    Message,
    tool,
)
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import Field
from rich import print
from rich.logging import RichHandler

# ── Logging ──────────────────────────────────────────────────────────
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Cliente de OpenAI ─────────────────────────────────────────────────
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
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4o"),
    )
else:
    client = OpenAIChatClient(api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-4o"))


# ── Tools ────────────────────────────────────────────────────────────


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Devuelve datos del clima para una ciudad."""
    conditions = ["soleado", "nublado", "lluvioso", "nevado"]
    temp = random.randint(30, 90)
    return f"El clima en {city} está {random.choice(conditions)} con una máxima de {temp}°F."


@tool
def get_activities(
    city: Annotated[str, Field(description="The city to find activities in.")],
) -> str:
    """Devuelve actividades populares de fin de semana para una ciudad."""
    all_activities = [
        "Visitar el mercado de agricultores",
        "Hacer senderismo en el parque estatal",
        "Ir a un festival de food trucks",
        "Ir al museo de arte",
        "Hacer un tour a pie por el centro",
        "Visitar el jardín botánico",
        "Ver un show de música en vivo",
        "Probar un brunch nuevo",
    ]
    picked = random.sample(all_activities, k=3)
    return f"Actividades populares en {city}: {', '.join(picked)}."


# ── Middleware de resumen ───────────────────────────────────────────

SUMMARIZE_PROMPT = (
    "Eres un asistente de resumen. Condensa la siguiente conversación "
    "en un resumen conciso que preserve todos los hechos clave, decisiones y contexto "
    "necesarios para continuar la conversación. Escribe el resumen en tercera persona. "
    "Sé conciso, pero no pierdas detalles importantes como ciudades específicas, "
    "condiciones del clima o recomendaciones que se hayan mencionado."
)


class SummarizationMiddleware(AgentMiddleware):
    """Middleware de agente que resume el historial cuando el uso de tokens supera un umbral.

    Implementa el patrón de "compactación de contexto": cuando el uso acumulado
    de tokens supera un umbral configurable, el middleware le pide al LLM que
    genere un resumen de la conversación y reemplaza los mensajes anteriores
    por ese resumen. Esto mantiene manejable la ventana de contexto en
    conversaciones largas.

    El middleware accede al historial de sesión mediante ``session.state`` (donde
    el ``InMemoryHistoryProvider`` integrado guarda mensajes) y lo reemplaza
    por un único mensaje de resumen antes de procesar el siguiente turno.
    """

    # Key used by the default InMemoryHistoryProvider to store messages
    # https://github.com/microsoft/agent-framework/issues/3941
    HISTORY_STATE_KEY = "memory"

    def __init__(
        self,
        client: OpenAIChatClient,
        token_threshold: int = 1000,
    ) -> None:
        """Inicializa el middleware de resumen.

        Args:
            client: Cliente LLM que se usa para generar resúmenes.
            token_threshold: Resumir cuando los tokens acumulados excedan este valor.
        """
        self.client = client
        self.token_threshold = token_threshold
        self.context_tokens = 0

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Convierte los mensajes en un bloque de texto para el resumidor."""
        lines: list[str] = []
        for msg in messages:
            if msg.text:
                lines.append(f"{msg.role}: {msg.text}")
        return "\n".join(lines)

    async def _summarize(self, messages: list[Message]) -> str:
        """Llama al LLM para resumir los mensajes de la conversación."""
        conversation_text = self._format_messages_for_summary(messages)
        summary_messages = [
            Message(role="system", text=SUMMARIZE_PROMPT),
            Message(role="user", text=f"Resume esta conversación:\n\n{conversation_text}"),
        ]
        response = await self.client.get_response(summary_messages)
        return response.text or "No hay resumen disponible."

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Revisa tokens y resume si pasa el umbral; luego continúa la ejecución."""
        session = context.session

        # Antes de ejecutar el agente: revisar si hay que compactar el historial
        if session and self.context_tokens > self.token_threshold:
            history = session.state.get(self.HISTORY_STATE_KEY, {}).get("messages", [])
            if len(history) > 2:
                logger.info(
                    "[📝 Resumen] Uso de tokens (%d) excede el umbral (%d). "
                    "Resumiendo %d mensajes del historial...",
                    self.context_tokens,
                    self.token_threshold,
                    len(history),
                )

                # Resumir el historial completo
                summary_text = await self._summarize(history)
                logger.info(
                    "[📝 Resumen] Resumen: %s",
                    summary_text[:200] + "..." if len(summary_text) > 200 else summary_text,
                )

                # Reemplazar el historial de la sesión con un único mensaje de resumen
                session.state[self.HISTORY_STATE_KEY]["messages"] = [
                    Message(role="assistant", text=f"[Resumen de la conversación anterior]\n{summary_text}"),
                ]

                # Reiniciar contador de tokens después del resumen
                self.context_tokens = 0
                logger.info("[📝 Resumen] Historial compactado a 1 mensaje de resumen")
        else:
            logger.info(
                "[📝 Resumen] Uso de tokens: %d / %d umbral. No hace falta resumir.",
                self.context_tokens,
                self.token_threshold,
            )

        # Ejecutar el agente (carga historial, llama al LLM y guarda la respuesta)
        await call_next()

        # Después: registrar uso de tokens de la respuesta
        if context.result and isinstance(context.result, AgentResponse) and context.result.usage_details:
            new_tokens = context.result.usage_details.get("total_token_count", 0) or 0
            self.context_tokens += new_tokens
            logger.info(
                "[📝 Resumen] Este turno usó %d tokens. Contexto: %d",
                new_tokens,
                self.context_tokens,
            )


# ── Configuración del agente ─────────────────────────────────────────

# Usar un umbral bajo para demo para que el resumen se dispare rápido
summarization_middleware = SummarizationMiddleware(client=client, token_threshold=500)

agent = Agent(
    name="weekend-planner",
    client=client,
    instructions=(
        "Eres un asistente útil para planear el fin de semana. Ayuda a la gente "
        "a planear su fin de semana revisando el clima y sugiriendo actividades. "
        "Sé amable y da recomendaciones detalladas."
    ),
    tools=[get_weather, get_activities],
    middleware=[summarization_middleware],
)


async def main() -> None:
    """Ejecuta una conversación multi-turno que dispara el resumen."""
    print("\n[bold]=== Compactación de contexto con resumen ===[/bold]")
    print(f"[dim]Umbral de tokens: {summarization_middleware.token_threshold}[/dim]")
    print("[dim]El middleware resumirá la conversación cuando el uso de tokens supere el umbral.[/dim]\n")

    session = agent.create_session()

    # Turno 1
    user_msg = "¿Cómo estará el clima en San Francisco este fin de semana?"
    print(f"[blue]Usuario:[/blue] {user_msg}")
    response = await agent.run(user_msg, session=session)
    print(f"[green]Agente:[/green] {response.text}\n")

    # Turno 2
    user_msg = "¿Y Portland? ¿Cómo estará el clima y qué actividades puedo hacer ahí?"
    print(f"[blue]Usuario:[/blue] {user_msg}")
    response = await agent.run(user_msg, session=session)
    print(f"[green]Agente:[/green] {response.text}\n")

    # Turno 3 — para este punto deberíamos estar cerca del umbral
    user_msg = "¿Qué tal Seattle? Dame el panorama completo: clima y cosas para hacer."
    print(f"[blue]Usuario:[/blue] {user_msg}")
    response = await agent.run(user_msg, session=session)
    print(f"[green]Agente:[/green] {response.text}\n")

    # Turno 4 — aquí debería dispararse el resumen
    user_msg = "De todas las ciudades que mencionamos, ¿cuál tiene la mejor combinación de clima y actividades?"
    print(f"[blue]Usuario:[/blue] {user_msg}")
    response = await agent.run(user_msg, session=session)
    print(f"[green]Agente:[/green] {response.text}\n")

    # Turno 5 — después del resumen, el agente debería conservar contexto
    user_msg = "Perfecto, vamos con esa ciudad. ¿Qué debería empacar?"
    print(f"[blue]Usuario:[/blue] {user_msg}")
    response = await agent.run(user_msg, session=session)
    print(f"[green]Agente:[/green] {response.text}\n")

    print(f"[dim]Conteo final de tokens en contexto: {summarization_middleware.context_tokens}[/dim]")

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
