"""Handoff orchestration with explicit routing rules (customer support).

Demonstrates: HandoffBuilder with .add_handoff() rules that enforce
business logic — e.g. triage cannot route directly to the refund agent;
only return_agent can escalate to refunds.

Routing rules:
    triage_agent  -> order_agent, return_agent   (NOT refund_agent)
    order_agent   -> triage_agent
    return_agent  -> triage_agent, refund_agent  (only path to refunds)
    refund_agent  -> triage_agent

Reference:
    https://learn.microsoft.com/agent-framework/workflows/orchestrations/handoff?pivots=programming-language-python#configure-handoff-rules-1

Run:
    uv run examples/workflow_handoffbuilder_rules.py
    uv run examples/workflow_handoffbuilder_rules.py --devui
"""

import asyncio
import logging
import os
import sys

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from rich.logging import RichHandler

log_handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[log_handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        model_id=os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
    )
else:
    client = OpenAIChatClient(
        api_key=os.environ["OPENAI_API_KEY"], model_id=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    )


# ── Agents ─────────────────────────────────────────────────────────────────

triage_agent = Agent(
    client=client,
    name="triage_agent",
    instructions=(
        "You are a customer-support triage agent. Greet the customer, understand their issue, "
        "and hand off to the right specialist: order_agent for order inquiries, "
        "return_agent for returns. You cannot handle refunds directly. "
        "When the conversation is resolved, say 'Goodbye!' to end the session."
    ),
)

order_agent = Agent(
    client=client,
    name="order_agent",
    instructions=(
        "You handle order status inquiries. Look up the customer's order and provide a brief update. "
        "When done, hand off back to triage_agent."
    ),
)

return_agent = Agent(
    client=client,
    name="return_agent",
    instructions=(
        "You handle product returns. Help the customer initiate a return. "
        "If they also want a refund, hand off to refund_agent. "
        "Otherwise, hand off back to triage_agent when done."
    ),
)

refund_agent = Agent(
    client=client,
    name="refund_agent",
    instructions=(
        "You process refunds for returned items. Confirm the refund details and let the "
        "customer know when to expect the money back. Hand off to triage_agent when done."
    ),
)

# ── Build the handoff workflow with explicit routing rules ─────────────────

workflow = (
    HandoffBuilder(
        name="customer_support_handoff",
        participants=[triage_agent, order_agent, return_agent, refund_agent],
        termination_condition=lambda conversation: (
            len(conversation) > 0 and "goodbye" in conversation[-1].text.lower()
        ),
    )
    .with_start_agent(triage_agent)
    # Triage cannot route directly to refund_agent
    .add_handoff(triage_agent, [order_agent, return_agent])
    # Only return_agent can escalate to refund_agent
    .add_handoff(return_agent, [refund_agent, triage_agent])
    # All specialists can hand back to triage
    .add_handoff(order_agent, [triage_agent])
    .add_handoff(refund_agent, [triage_agent])
    .with_autonomous_mode()
    .build()
)


async def main() -> None:
    """Run a customer support handoff workflow with explicit routing rules."""
    request = "I want to return a jacket I bought last week and get a refund."
    logger.info("Request: %s\n", request)

    result = await workflow.run(request)
    print(result[-1].text)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8098, auto_open=True)
    else:
        asyncio.run(main())
