"""Fan-out/fan-in with LLM-as-judge ranking aggregation.

Three creative agents with different personas (bold, minimalist,
emotional) each propose a marketing slogan.  A formatter collects the
candidates, then a judge Agent scores and ranks them — letting the LLM
evaluate creativity, memorability, and brand fit.

Aggregation technique: LLM-as-judge (generate N candidates, rank the best).

Run:
    uv run examples/workflow_aggregator_ranked.py
    uv run examples/workflow_aggregator_ranked.py --devui  (opens DevUI at http://localhost:8104)
"""

import asyncio
import os
import sys

from agent_framework import Agent, AgentExecutorResponse, Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from typing_extensions import Never

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

# Configure the chat client based on the API host
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


class DispatchPrompt(Executor):
    """Emit the product brief downstream for fan-out broadcast."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(prompt)


class FormatCandidates(Executor):
    """Fan-in aggregator that formats candidate slogans for the judge."""

    @handler
    async def format(
        self,
        results: list[AgentExecutorResponse],
        ctx: WorkflowContext[Never, str],
    ) -> None:
        """Collect slogans into a labeled list for the judge Agent."""
        lines = []
        for result in results:
            slogan = result.agent_response.text.strip().strip("\"'").split("\n")[0].strip().strip("\"'")
            lines.append(f"- [{result.executor_id}]: \"{slogan}\"")
        await ctx.send_message("Candidate slogans:\n" + "\n".join(lines))


dispatcher = DispatchPrompt(id="dispatcher")

bold_writer = Agent(
    client=client,
    name="BoldWriter",
    instructions=(
        "You are a bold, dramatic copywriter. "
        "Given the product brief, propose ONE punchy marketing slogan (max 10 words). "
        "Make it attention-grabbing and confident. Reply with ONLY the slogan."
    ),
)

minimalist_writer = Agent(
    client=client,
    name="MinimalistWriter",
    instructions=(
        "You are a minimalist copywriter who values brevity above all. "
        "Given the product brief, propose ONE ultra-short marketing slogan (max 6 words). "
        "Less is more. Reply with ONLY the slogan."
    ),
)

emotional_writer = Agent(
    client=client,
    name="EmotionalWriter",
    instructions=(
        "You are an empathy-driven copywriter. "
        "Given the product brief, propose ONE marketing slogan (max 10 words) "
        "that connects emotionally with the audience. Reply with ONLY the slogan."
    ),
)

ranker = FormatCandidates(id="ranker")

judge = Agent(
    client=client,
    name="Judge",
    instructions=(
        "You are a senior creative director judging marketing slogans. "
        "Given a list of candidate slogans, rank them from best to worst. "
        "For each slogan, give a 1-10 score and a one-sentence justification "
        "evaluating creativity, memorability, clarity, and brand fit. "
        "Format: #1 (score X) [AgentName]: \"slogan\" — justification"
    ),
)

workflow = (
    WorkflowBuilder(
        name="FanOutFanInRanked",
        description="Generate slogans in parallel, then LLM-judge ranks them.",
        start_executor=dispatcher,
        output_executors=[judge],
    )
    .add_fan_out_edges(dispatcher, [bold_writer, minimalist_writer, emotional_writer])
    .add_fan_in_edges([bold_writer, minimalist_writer, emotional_writer], ranker)
    .add_edge(ranker, judge)
    .build()
)


async def main() -> None:
    """Run the slogan pipeline and print the ranked results."""
    prompt = "Budget-friendly electric bike for urban commuters. Reliable, affordable, green."
    print(f"Product brief: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        print(output)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8104, auto_open=True)
    else:
        asyncio.run(main())
