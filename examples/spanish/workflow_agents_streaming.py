"""Workflow Escritor → Revisor con eventos de streaming.

Demuestra: run(stream=True) para consumir eventos del workflow en tiempo real,
incluyendo executor_invoked, executor_completed y tokens de salida en streaming.

Tipos de eventos observados:
  "started"             — comienza la ejecución del workflow
  "executor_invoked"   — un ejecutor (agente) comienza a procesar
  "output"             — un fragmento de texto en streaming (AgentResponseUpdate)
  "executor_completed" — un ejecutor termina
  "executor_failed"    — un ejecutor encuentra un error
  "error"              — el workflow encuentra un error
  "warning"            — el workflow encontró una advertencia

Contrasta con workflow_agents.py, que usa run() e imprime solo la salida final.

Referencia:
    https://learn.microsoft.com/en-us/agent-framework/workflows/events?pivots=programming-language-python

Ejecutar:
    uv run examples/spanish/workflow_agents_streaming.py
"""

import asyncio
import os

from agent_framework import Agent, AgentResponseUpdate, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

# Configura el cliente según el host de la API
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
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4o-mini"),
    )
else:
    client = OpenAIChatClient(
        api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    )

writer = Agent(
    client=client,
    name="Escritor",
    instructions=(
        "Eres un escritor de contenido conciso. "
        "Escribe un artículo corto (2-3 párrafos) claro y atractivo sobre el tema indicado. "
        "Prioriza la precisión y la legibilidad."
    ),
)

reviewer = Agent(
    client=client,
    name="Revisor",
    instructions=(
        "Eres un revisor de contenido reflexivo. "
        "Lee el borrador del escritor y ofrece retroalimentación específica y constructiva. "
        "Comenta sobre la claridad, la precisión y la estructura. Mantén tu revisión concisa."
    ),
)

workflow = WorkflowBuilder(name="EscritorRevisor", start_executor=writer).add_edge(writer, reviewer).build()


async def main():
    prompt = "Escribe una publicación corta de LinkedIn: \"4 trabajos que los agentes de IA están transformando silenciosamente este año.\""
    print(f"💬 Solicitud: {prompt}\n")

    async for event in workflow.run(prompt, stream=True):
        if event.type == "started":
            print(f"📡 Evento started | workflow={workflow.name}")
        elif event.type == "executor_invoked":
            print(f"\n📡 Evento executor_invoked | executor={event.executor_id}")
        elif event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            print(event.data.text, end="", flush=True)
        elif event.type == "executor_completed":
            print(f"\n\n📡 Evento executor_completed | executor={event.executor_id}")
        elif event.type == "executor_failed":
            print(f"\n📡 Evento executor_failed | executor={event.executor_id} | details={event.data}")
        elif event.type == "error":
            print(f"\n📡 Evento error | details={event.data}")
        elif event.type == "warning":
            print(f"\n📡 Evento warning | details={event.data}")

    print()

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
