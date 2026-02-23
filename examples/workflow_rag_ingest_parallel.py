"""RAG ingestion pipeline with explicit parallel fan-out/fan-in.

Demonstrates: data-level parallelism in WorkflowBuilder using
add_fan_out_edges + add_fan_in_edges.

Pipeline:
    Extract -> Chunk -> [EmbedWorker0, EmbedWorker1, EmbedWorker2] -> Fan-in Merge

Run:
    uv run examples/workflow_rag_ingest_parallel.py
    uv run examples/workflow_rag_ingest_parallel.py --devui  (opens DevUI at http://localhost:8098)
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI
from rich.logging import RichHandler
from typing_extensions import Never

log_handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[log_handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
EMBEDDING_DIMENSIONS = 256
WORKER_COUNT = 3

# Configure the embedding client based on the API host
if API_HOST == "azure":
    sync_credential = DefaultAzureCredential()
    sync_token_provider = get_bearer_token_provider(sync_credential, "https://cognitiveservices.azure.com/.default")
    embed_client = OpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
        api_key=sync_token_provider(),
    )
    embed_model = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
elif API_HOST == "github":
    embed_client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
    )
    embed_model = "text-embedding-3-small"
else:
    embed_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embed_model = "text-embedding-3-small"


@dataclass
class EmbeddedChunk:
    """A text chunk with source index and embedding vector."""

    index: int
    text: str
    vector: list[float] = field(default_factory=list)


class ExtractExecutor(Executor):
    """Convert an input file to markdown text."""

    @handler
    async def extract(self, path: str, ctx: WorkflowContext[str]) -> None:
        """Convert a file path to markdown for downstream chunking."""
        path = path.strip("'\"")
        resolved = Path(path) if Path(path).is_absolute() else Path(__file__).parent.parent / path
        result = MarkItDown().convert(str(resolved))
        await ctx.send_message(result.text_content)


class ChunkExecutor(Executor):
    """Split markdown into substantive chunks."""

    @handler
    async def chunk(self, markdown: str, ctx: WorkflowContext[list[str]]) -> None:
        """Create a list of chunk strings for parallel embedding."""
        paragraphs = markdown.split("\n\n")
        chunks = [p.strip() for p in paragraphs if len(p.strip()) >= 80 and not p.strip().startswith("#")]
        logger.info("→ %d chunks extracted", len(chunks))
        await ctx.send_message(chunks)


class EmbedWorkerExecutor(Executor):
    """Embed a deterministic shard of chunks based on worker index."""

    def __init__(self, worker_index: int, id: str) -> None:
        """Initialize one worker with a fixed modulo shard assignment."""
        super().__init__(id=id)
        self.worker_index = worker_index

    @handler
    async def embed(self, chunks: list[str], ctx: WorkflowContext[list[EmbeddedChunk]]) -> None:
        """Embed only chunks assigned to this worker and send partial results."""
        embedded: list[EmbeddedChunk] = []
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index % WORKER_COUNT != self.worker_index:
                continue
            response = embed_client.embeddings.create(input=chunk, model=embed_model, dimensions=EMBEDDING_DIMENSIONS)
            embedded.append(
                EmbeddedChunk(index=chunk_index, text=chunk, vector=response.data[0].embedding),
            )

        logger.info("→ %s embedded %d chunks", self.id, len(embedded))
        await ctx.send_message(embedded)


class MergeEmbeddingsExecutor(Executor):
    """Fan-in reducer that merges partial embedding lists from all workers."""

    @handler
    async def merge(
        self,
        partial_results: list[list[EmbeddedChunk]],
        ctx: WorkflowContext[Never, list[EmbeddedChunk]],
    ) -> None:
        """Flatten worker results, restore original order, and output combined embeddings."""
        merged: list[EmbeddedChunk] = []
        for partial in partial_results:
            merged.extend(partial)

        merged.sort(key=lambda item: item.index)
        logger.info("→ fan-in merged %d total embedded chunks", len(merged))
        await ctx.yield_output(merged)


extract = ExtractExecutor(id="extract")
chunk = ChunkExecutor(id="chunk")
embed_worker_0 = EmbedWorkerExecutor(worker_index=0, id="embed_worker_0")
embed_worker_1 = EmbedWorkerExecutor(worker_index=1, id="embed_worker_1")
embed_worker_2 = EmbedWorkerExecutor(worker_index=2, id="embed_worker_2")
merge = MergeEmbeddingsExecutor(id="merge")

workflow = (
    WorkflowBuilder(
        name="RagIngestParallel",
        description="RAG ingest with explicit fan-out/fan-in parallel embedding.",
        start_executor=extract,
    )
    .add_edge(extract, chunk)
    .add_fan_out_edges(chunk, [embed_worker_0, embed_worker_1, embed_worker_2])
    .add_fan_in_edges([embed_worker_0, embed_worker_1, embed_worker_2], merge)
    .build()
)


async def main() -> None:
    """Run the workflow on sample PDF and print embedding preview."""
    pdf_path = str(Path(__file__).parent / "sample_document.pdf")
    logger.info("Processing: %s", pdf_path)

    events = await workflow.run(pdf_path)
    outputs = events.get_outputs()

    for embedded_chunks in outputs:
        logger.info("Embedded %d chunks (parallel fan-out/fan-in)", len(embedded_chunks))
        for item in embedded_chunks[:5]:
            preview = item.text[:80].replace("\n", " ")
            logger.info("  [idx=%d, %dd] %s…", item.index, len(item.vector), preview)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8098, auto_open=True)
    else:
        asyncio.run(main())
