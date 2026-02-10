"""
OpenTelemetry + Aspire Dashboard example.

Demonstrates a tool-calling agent that exports OpenTelemetry traces, metrics,
and structured logs to the .NET Aspire Dashboard via OTLP/gRPC.

Telemetry is only exported when the OTEL_EXPORTER_OTLP_ENDPOINT environment
variable is set. Without it, the agent runs normally with no telemetry export.

To start the Aspire Dashboard:

    docker run --rm -it -d \
        -p 18888:18888 \
        -p 4317:18889 \
        --name aspire-dashboard \
        mcr.microsoft.com/dotnet/aspire-dashboard:latest

The dashboard UI is at http://localhost:18888.
Get the login token from the container logs:

    docker logs aspire-dashboard

Look for: "Login to the dashboard at http://localhost:18888/login?t=<TOKEN>"

Then run this example with telemetry export enabled:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python examples/agent_otel_aspire.py

In the Aspire Dashboard you will see:
  - Traces:  agent -> chat completion -> tool execution spans
  - Metrics: token usage and operation duration histograms
  - Structured Logs: conversation messages (system, user, assistant, tool)
  - GenAI telemetry visualizer: full conversation view on chat spans

To stop the dashboard:

    docker stop aspire-dashboard

For the full Python + Aspire guide, see:
  https://aspire.dev/dashboard/standalone-for-python/
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.observability import configure_otel_providers
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

# Configure OpenTelemetry export to the Aspire Dashboard (if endpoint is set)
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    os.environ.setdefault("OTEL_SERVICE_NAME", "agent-framework-demo")
    configure_otel_providers(enable_sensitive_data=True)
    logger.info(f"OpenTelemetry export enabled — sending to {otlp_endpoint}")
else:
    logger.info("Set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 to export telemetry to the Aspire Dashboard")

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


def get_weather(
    city: Annotated[str, Field(description="City name, spelled out fully")],
) -> dict:
    """Returns weather data for a given city, a dictionary with temperature and description."""
    logger.info(f"Getting weather for {city}")
    weather_options = [
        {"temperature": 72, "description": "Sunny"},
        {"temperature": 60, "description": "Rainy"},
        {"temperature": 55, "description": "Cloudy"},
        {"temperature": 45, "description": "Windy"},
    ]
    return random.choice(weather_options)


def get_current_time(
    timezone_name: Annotated[str, Field(description="Timezone name, e.g. 'US/Eastern', 'Asia/Tokyo', 'UTC'")],
) -> str:
    """Returns the current date and time in UTC (timezone_name is for display context only)."""
    logger.info(f"Getting current time for {timezone_name}")
    now = datetime.now(timezone.utc)
    return f"The current time in {timezone_name} is approximately {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"


agent = ChatAgent(
    name="weather-time-agent",
    chat_client=client,
    instructions="You are a helpful assistant that can look up weather and time information.",
    tools=[get_weather, get_current_time],
)


async def main():
    response = await agent.run("What's the weather in Seattle and what time is it in Tokyo?")
    print(response.text)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
