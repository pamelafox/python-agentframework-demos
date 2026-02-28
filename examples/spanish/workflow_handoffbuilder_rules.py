"""Orquestación de handoff con reglas explícitas de ruteo (soporte al cliente).

Demuestra: HandoffBuilder con reglas .add_handoff() que aplican lógica de
negocio — por ejemplo, triage no puede rutear directo al agente de reembolsos;
solo return_agent puede escalar a refunds.

Reglas de ruteo:
    triage_agent  -> order_agent, return_agent   (NO refund_agent)
    order_agent   -> triage_agent
    return_agent  -> triage_agent, refund_agent  (único camino a refunds)
    refund_agent  -> triage_agent

Referencia:
    https://learn.microsoft.com/agent-framework/workflows/orchestrations/handoff?pivots=programming-language-python#configure-handoff-rules-1

Ejecutar:
    uv run examples/spanish/workflow_handoffbuilder_rules.py
    uv run examples/spanish/workflow_handoffbuilder_rules.py --devui
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

# Configura el cliente de chat según el proveedor de API
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


# ── Agentes ───────────────────────────────────────────────────────────────

triage_agent = Agent(
    client=client,
    name="triage_agent",
    instructions=(
        "Eres un agente de triage de soporte al cliente. Saluda al cliente, entiende su problema "
        "y haz handoff al especialista correcto: order_agent para temas de pedidos y "
        "return_agent para devoluciones. No puedes gestionar reembolsos directamente. "
        "Cuando el caso esté resuelto, di 'Goodbye!' (Goodbye/Adiós) para terminar la sesión."
    ),
)

order_agent = Agent(
    client=client,
    name="order_agent",
    instructions=(
        "Atiendes consultas sobre el estado del pedido. Busca el pedido del cliente y da una actualización breve. "
        "Al terminar, haz handoff de vuelta a triage_agent."
    ),
)

return_agent = Agent(
    client=client,
    name="return_agent",
    instructions=(
        "Atiendes devoluciones de productos. Ayuda al cliente a iniciar una devolución. "
        "Si también quiere un reembolso, haz handoff a refund_agent. "
        "De lo contrario, haz handoff de vuelta a triage_agent al terminar."
    ),
)

refund_agent = Agent(
    client=client,
    name="refund_agent",
    instructions=(
        "Procesas reembolsos por artículos devueltos. Confirma los detalles del reembolso y avísale "
        "al cliente cuándo puede esperar el dinero de vuelta. Haz handoff a triage_agent al terminar."
    ),
)

# ── Construye el workflow de handoff con reglas explícitas ─────────────────

workflow = (
    HandoffBuilder(
        name="customer_support_handoff",
        participants=[triage_agent, order_agent, return_agent, refund_agent],
        termination_condition=lambda conversation: (
            len(conversation) > 0 and "goodbye" in conversation[-1].text.lower()
        ),
    )
    .with_start_agent(triage_agent)
    # triage_agent no puede rutear directamente a refund_agent
    .add_handoff(triage_agent, [order_agent, return_agent])
    # Solo return_agent puede escalar a refund_agent
    .add_handoff(return_agent, [refund_agent, triage_agent])
    # Todos los especialistas pueden regresar a triage
    .add_handoff(order_agent, [triage_agent])
    .add_handoff(refund_agent, [triage_agent])
    .with_autonomous_mode()
    .build()
)


async def main() -> None:
    """Ejecuta un workflow de soporte con handoff y reglas explícitas de ruteo."""
    request = "Quiero devolver una chamarra que compré la semana pasada y recibir un reembolso."
    logger.info("Solicitud: %s\n", request)

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
