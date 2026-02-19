"""
Context isolation with sub-agents.

When an agent delegates tool-heavy work to a sub-agent, the sub-agent's
context window absorbs all the raw tool output (file contents, search
results, etc.). The main agent only sees the sub-agent's concise summary,
keeping its own context window small and focused.

This is the "context quarantine" pattern described in:
- LangChain deep agents: https://docs.langchain.com/oss/python/deepagents/subagents
- Manus context engineering: https://rlancemartin.github.io/2025/10/15/manus/
- Google ADK architecture: https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools/
- VS Code subagents: https://code.visualstudio.com/docs/copilot/agents/subagents

Diagram:

 agent.run("user question")
 │
 ▼
 ┌─────────────────────────────────────────────────────────┐
 │              Coordinator                                │
 │  (small context — only sees summaries)                  │
 │                                                         │
 │  Calls research_codebase("question")                    │
 │       │                                                 │
 │       ▼                                                 │
 │  ┌──────────────────────────────────────────────────┐   │
 │  │         Research Sub-Agent                       │   │
 │  │  (isolated context — absorbs all raw content)    │   │
 │  │                                                  │   │
 │  │  1. list_project_files() → file listing          │   │
 │  │  2. read_project_file() → full file contents     │   │
 │  │  3. search_project_files() → matching lines      │   │
 │  │  4. Returns concise summary (< 200 words)        │   │
 │  └──────────────────────────────────────────────────┘   │
 │       │                                                 │
 │       ▼ summary text only                               │
 │  Synthesizes final answer from summary                  │
 └─────────────────────────────────────────────────────────┘
 │
 ▼
 response (coordinator never saw raw file contents)

Compare with agent_without_subagent.py to see the difference.
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

# ── OpenAI client ────────────────────────────────────────────────────
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

# ── Project root for file tools ──────────────────────────────────────
PROJECT_DIR = os.path.join(os.path.dirname(__file__))

# ── Sub-agent token tracking ─────────────────────────────────────────
# We accumulate sub-agent usage here so main() can report it alongside
# the coordinator's usage for a side-by-side comparison.
subagent_usage_log: list[dict] = []


# ── File tools (given to the research sub-agent only) ────────────────

@tool
def list_project_files(
    directory: Annotated[str, Field(description="Relative directory path within the examples folder, e.g. '.' or 'spanish'.")],
) -> str:
    """List all files in the given directory under the examples folder."""
    logger.info("[📂 Tool] list_project_files('%s')", directory)
    target = os.path.join(PROJECT_DIR, directory)
    if not os.path.isdir(target):
        return f"Error: directory '{directory}' not found."
    entries = sorted(os.listdir(target))
    return "\n".join(entries)


@tool
def read_project_file(
    filepath: Annotated[str, Field(description="Relative file path within the examples folder, e.g. 'agent_middleware.py'.")],
) -> str:
    """Read and return the full contents of a file in the examples folder."""
    logger.info("[📄 Tool] read_project_file('%s')", filepath)
    target = os.path.join(PROJECT_DIR, filepath)
    if not os.path.isfile(target):
        return f"Error: file '{filepath}' not found."
    with open(target) as f:
        return f.read()


@tool
def search_project_files(
    query: Annotated[str, Field(description="Text to search for (case-insensitive) across all .py files in the examples folder.")],
) -> str:
    """Search all .py files in the examples folder for lines containing the query string."""
    logger.info("[🔍 Tool] search_project_files('%s')", query)
    query_lower = query.lower()
    results: list[str] = []
    for fpath in sorted(glob.glob(os.path.join(PROJECT_DIR, "*.py"))):
        relpath = os.path.relpath(fpath, PROJECT_DIR)
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                if query_lower in line.lower():
                    results.append(f"{relpath}:{lineno}: {line.rstrip()}")
    if not results:
        return f"No matches found for '{query}'."
    if len(results) > 50:
        return "\n".join(results[:50]) + f"\n... ({len(results) - 50} more matches truncated)"
    return "\n".join(results)


# ── Research sub-agent ───────────────────────────────────────────────
# This agent gets the file tools and reads raw source code.
# Its context window absorbs all the verbose tool output.

research_agent = Agent(
    name="research-agent",
    client=client,
    instructions=(
        "You are a code research assistant. Use the available tools to list, "
        "read, and search Python source files in the project to answer the "
        "question. Be thorough in your research but return a CONCISE summary "
        "of your findings in under 200 words. Do NOT include raw file "
        "contents in your response — summarize the key patterns, classes, "
        "and functions you found."
    ),
    tools=[list_project_files, read_project_file, search_project_files],
)


# ── Delegation tool (given to the coordinator) ───────────────────────


@tool
async def research_codebase(
    question: Annotated[str, Field(description="A research question about the codebase to investigate.")],
) -> str:
    """Delegate a code research question to the research sub-agent.

    The sub-agent reads and searches files in its own isolated context,
    then returns a concise summary. The coordinator never sees the raw
    file contents.
    """
    logger.info("[🔬 Sub-Agent] Delegating: %s", question[:80])

    response = await research_agent.run(question)

    # Track sub-agent token usage for comparison
    usage = response.usage_details or {}
    subagent_usage_log.append(usage)
    input_t = usage.get("input_token_count", 0) or 0
    output_t = usage.get("output_token_count", 0) or 0
    total_t = usage.get("total_token_count", 0) or 0
    logger.info(
        "[🔬 Sub-Agent] Done. Sub-agent used input=%d output=%d total=%d tokens",
        input_t, output_t, total_t,
    )

    return response.text or "No findings."


# ── Coordinator agent ────────────────────────────────────────────────
# This agent only has the research_codebase tool — it never sees raw
# file contents. Its context window stays small and focused.

coordinator = Agent(
    name="coordinator",
    client=client,
    instructions=(
        "You are a helpful coding assistant. You answer questions about "
        "codebases, explain patterns, and help developers understand code. "
        "Use the research_codebase tool to investigate the codebase before "
        "answering — it will read and search files for you. Provide a "
        "clear, well-organized answer based on the research results."
    ),
    tools=[research_codebase],
)

# ── Query ────────────────────────────────────────────────────────────

USER_QUERY = "What different middleware patterns are used across this project? Read the relevant files to find out."


async def main() -> None:
    """Run a single query and compare coordinator vs sub-agent token usage."""
    print("\n[bold]=== Code Research WITH Sub-Agents (Context Isolation) ===[/bold]")
    print("[dim]The coordinator delegates file reading to a research sub-agent.[/dim]")
    print("[dim]Raw file contents stay in the sub-agent's context, not the coordinator's.[/dim]\n")

    subagent_usage_log.clear()

    print(f"[blue]User:[/blue] {USER_QUERY}")
    response = await coordinator.run(USER_QUERY)
    print(f"[green]Coordinator:[/green] {response.text}\n")

    # Coordinator token usage
    coord_usage = response.usage_details or {}
    coord_input = coord_usage.get("input_token_count", 0) or 0
    coord_output = coord_usage.get("output_token_count", 0) or 0
    coord_total = coord_usage.get("total_token_count", 0) or 0

    # Sub-agent token usage (accumulated across all delegation calls)
    sub_input = sum((u.get("input_token_count", 0) or 0) for u in subagent_usage_log)
    sub_output = sum((u.get("output_token_count", 0) or 0) for u in subagent_usage_log)
    sub_total = sum((u.get("total_token_count", 0) or 0) for u in subagent_usage_log)

    print("[bold]── Token Usage ──[/bold]")
    print(f"[yellow]  Coordinator tokens:[/yellow]  input={coord_input:,}  output={coord_output:,}  total={coord_total:,}")
    print(f"[yellow]  Sub-agent tokens:[/yellow]  input={sub_input:,}  output={sub_output:,}  total={sub_total:,}")
    print()
    print("[dim]The coordinator's input tokens are much lower because it never saw[/dim]")
    print("[dim]raw file contents — only the sub-agent's concise summary.[/dim]")
    print("[dim]Compare with agent_without_subagent.py where ALL file contents are in context.[/dim]\n")

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[coordinator], auto_open=True)
    else:
        asyncio.run(main())
