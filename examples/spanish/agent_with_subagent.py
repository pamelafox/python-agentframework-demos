"""Aislamiento de contexto con subagentes.

Cuando un agente delega trabajo pesado de herramientas a un subagente, la
ventana de contexto del subagente absorbe toda la salida cruda de herramientas
(contenido de archivos, resultados de búsqueda, etc.). El agente principal solo
ve el resumen conciso del subagente, manteniendo su propia ventana de contexto
chica y enfocada.

Este es el patrón de "cuarentena de contexto" descrito en:
- LangChain deep agents: https://docs.langchain.com/oss/python/deepagents/subagents
- Manus context engineering: https://rlancemartin.github.io/2025/10/15/manus/
- Google ADK architecture: https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools/
- VS Code subagents: https://code.visualstudio.com/docs/copilot/agents/subagents

Diagrama:

 agent.run("pregunta del usuario")
 │
 ▼
 ┌─────────────────────────────────────────────────────────┐
 │              Coordinador                                │
 │  (contexto pequeño — solo ve resúmenes)                 │
 │                                                         │
 │  Llama a research_codebase("pregunta")                  │
 │       │                                                 │
 │       ▼                                                 │
 │  ┌──────────────────────────────────────────────────┐   │
 │  │         Subagente de Investigación               │   │
 │  │  (contexto aislado — absorbe contenido crudo)    │   │
 │  │                                                  │
 │  │  1. list_project_files() → lista de archivos      │   │
 │  │  2. read_project_file() → contenido completo      │   │
 │  │  3. search_project_files() → líneas coincidentes  │   │
 │  │  4. Regresa un resumen conciso (< 200 palabras)   │   │
 │  └──────────────────────────────────────────────────┘   │
 │       │                                                 │
 │       ▼ solo texto resumen                               │
 │  Sintetiza la respuesta final desde el resumen          │
 └─────────────────────────────────────────────────────────┘
 │
 ▼
 respuesta (el coordinador nunca vio contenido crudo)

Compara con agent_without_subagent.py para ver la diferencia.
"""

import asyncio
import glob
import logging
import os
import sys
from typing import Annotated

from agent_framework import Agent, tool
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

# ── Cliente OpenAI ───────────────────────────────────────────────────
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

# ── Raíz del proyecto para herramientas de archivos ──────────────────
PROJECT_DIR = os.path.join(os.path.dirname(__file__))

# ── Seguimiento de tokens del subagente ──────────────────────────────
# Acumulamos el uso del subagente aquí para que main() lo pueda reportar
# junto con el uso del coordinador y comparar lado a lado.
subagent_usage_log: list[dict] = []


# ── Herramientas de archivos (solo para el subagente) ────────────────

@tool
def list_project_files(
    directory: Annotated[str, Field(description="Relative directory path within the examples folder, e.g. '.' or 'spanish'.")],
) -> str:
    """Lista todos los archivos en el directorio indicado dentro de la carpeta examples."""
    logger.info("[📂 Herramienta] list_project_files('%s')", directory)
    target = os.path.join(PROJECT_DIR, directory)
    if not os.path.isdir(target):
        return f"Error: no se encontró el directorio '{directory}'."
    entries = sorted(os.listdir(target))
    return "\n".join(entries)


@tool
def read_project_file(
    filepath: Annotated[str, Field(description="Relative file path within the examples folder, e.g. 'agent_middleware.py'.")],
) -> str:
    """Lee y devuelve el contenido completo de un archivo dentro de la carpeta examples."""
    logger.info("[📄 Herramienta] read_project_file('%s')", filepath)
    target = os.path.join(PROJECT_DIR, filepath)
    if not os.path.isfile(target):
        return f"Error: no se encontró el archivo '{filepath}'."
    with open(target) as f:
        return f.read()


@tool
def search_project_files(
    query: Annotated[str, Field(description="Text to search for (case-insensitive) across all .py files in the examples folder.")],
) -> str:
    """Busca en todos los .py dentro de examples las líneas que contengan el texto."""
    logger.info("[🔍 Herramienta] search_project_files('%s')", query)
    query_lower = query.lower()
    results: list[str] = []
    for fpath in sorted(glob.glob(os.path.join(PROJECT_DIR, "*.py"))):
        relpath = os.path.relpath(fpath, PROJECT_DIR)
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                if query_lower in line.lower():
                    results.append(f"{relpath}:{lineno}: {line.rstrip()}")
    if not results:
        return f"No se encontraron coincidencias para '{query}'."
    if len(results) > 50:
        return "\n".join(results[:50]) + f"\n... ({len(results) - 50} coincidencias más truncadas)"
    return "\n".join(results)


# ── Subagente de investigación ───────────────────────────────────────
# Este agente tiene las herramientas de archivos y lee el código fuente.
# Su ventana de contexto absorbe toda la salida verbosa de herramientas.

research_agent = Agent(
    name="research-agent",
    client=client,
    instructions=(
        "Eres un asistente de investigación de código. Usa las herramientas disponibles "
        "para listar, leer y buscar archivos Python del proyecto para responder la "
        "pregunta. Sé minucioso en tu investigación, pero regresa un resumen CONCISO "
        "de tus hallazgos en menos de 200 palabras. NO incluyas el contenido crudo de "
        "archivos en tu respuesta — resume los patrones, clases y funciones clave que encontraste."
    ),
    tools=[list_project_files, read_project_file, search_project_files],
)


# ── Herramienta de delegación (para el coordinador) ──────────────────


@tool
async def research_codebase(
    question: Annotated[str, Field(description="A research question about the codebase to investigate.")],
) -> str:
    """Delega una pregunta de investigación de código al subagente.

    El subagente lee y busca archivos en su propio contexto aislado, y luego
    regresa un resumen conciso. El coordinador nunca ve el contenido crudo.
    """
    logger.info("[🔬 Subagente] Delegando: %s", question[:80])

    response = await research_agent.run(question)

    # Registrar el uso de tokens del subagente para comparar
    usage = response.usage_details or {}
    subagent_usage_log.append(usage)
    input_t = usage.get("input_token_count", 0) or 0
    output_t = usage.get("output_token_count", 0) or 0
    total_t = usage.get("total_token_count", 0) or 0
    logger.info(
        "[🔬 Subagente] Listo. El subagente usó input=%d output=%d total=%d tokens",
        input_t, output_t, total_t,
    )

    return response.text or "No findings."


# ── Agente coordinador ───────────────────────────────────────────────
# Este agente solo tiene la herramienta research_codebase — nunca ve el contenido
# crudo de archivos. Su ventana de contexto se mantiene chica y enfocada.

coordinator = Agent(
    name="coordinator",
    client=client,
    instructions=(
        "Eres un asistente de programación útil. Respondes preguntas sobre bases de código, "
        "explicas patrones y ayudas a devs a entender el código. Usa la herramienta "
        "research_codebase para investigar la base de código antes de responder: va a leer "
        "y buscar archivos por ti. Da una respuesta clara y bien organizada basada en los resultados."
    ),
    tools=[research_codebase],
)

# ── Query ────────────────────────────────────────────────────────────

USER_QUERY = "¿Qué patrones distintos de middleware se usan en este proyecto? Lee los archivos relevantes para averiguarlo."


async def main() -> None:
    """Ejecuta una consulta y compara tokens del coordinador vs subagente."""
    print("\n[bold]=== Investigación de Código CON Subagentes (Aislamiento de Contexto) ===[/bold]")
    print("[dim]El coordinador delega la lectura de archivos a un subagente de investigación.[/dim]")
    print("[dim]El contenido crudo se queda en el contexto del subagente, no en el del coordinador.[/dim]\n")

    subagent_usage_log.clear()

    print(f"[blue]Usuario:[/blue] {USER_QUERY}")
    response = await coordinator.run(USER_QUERY)
    print(f"[green]Coordinador:[/green] {response.text}\n")

    # Uso de tokens del coordinador
    coord_usage = response.usage_details or {}
    coord_input = coord_usage.get("input_token_count", 0) or 0
    coord_output = coord_usage.get("output_token_count", 0) or 0
    coord_total = coord_usage.get("total_token_count", 0) or 0

    # Uso de tokens del subagente (acumulado entre llamadas de delegación)
    sub_input = sum((u.get("input_token_count", 0) or 0) for u in subagent_usage_log)
    sub_output = sum((u.get("output_token_count", 0) or 0) for u in subagent_usage_log)
    sub_total = sum((u.get("total_token_count", 0) or 0) for u in subagent_usage_log)

    print("[bold]── Uso de tokens ──[/bold]")
    print(f"[yellow]  Tokens del coordinador:[/yellow]  input={coord_input:,}  output={coord_output:,}  total={coord_total:,}")
    print(f"[yellow]  Tokens del subagente:[/yellow]  input={sub_input:,}  output={sub_output:,}  total={sub_total:,}")
    print()
    print("[dim]Los tokens de entrada del coordinador son mucho menores porque nunca vio[/dim]")
    print("[dim]contenido crudo — solo el resumen conciso del subagente.[/dim]")
    print("[dim]Compara con agent_without_subagent.py donde TODO el contenido queda en el contexto.[/dim]\n")

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[coordinator], auto_open=True)
    else:
        asyncio.run(main())
