# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Annotated

from agent_framework import (
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentResponse,
    AgentRunUpdateEvent,
    ChatAgent,
    ChatMessage,
    Content,
    Executor,
    RequestInfoEvent,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
    response_handler,
    tool,
)
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import Never

# Configura el cliente para usar Azure OpenAI, GitHub Models u OpenAI
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

"""
Example: agents with tools and human feedback

Pipeline design:
writer_agent (uses Azure OpenAI tools) -> Coordinator -> writer_agent
-> Coordinator -> final_editor_agent -> Coordinator -> output

The writer agent calls tools to gather product data before drafting.
A custom executor packages the draft and emits a RequestInfoEvent so a human can comment;
it then incorporates that guidance into the conversation before the final editor produces the polished output.

Demonstrates:
- Attaching tools (Python functions) to an agent inside a workflow.
- Capturing the writer output for human review.
- Streaming AgentRunUpdateEvent updates alongside human-in-the-loop pauses.

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Run `az login` before running.
"""


# NOTA: approval_mode="never_require" es por brevedad del ejemplo.
@tool(approval_mode="never_require")
def fetch_product_brief(
    product_name: Annotated[str, Field(description="Product name to look up.")],
) -> str:
    """Return a marketing brief for a product."""
    briefs = {
        "lumenx desk lamp": (
            "Producto: Lámpara de Escritorio LumenX\n"
            "- Brazo ajustable de tres puntos con rotación de 270°.\n"
            "- Espectro LED personalizado de cálido a neutro (2700K-4000K).\n"
            "- Almohadilla de carga USB-C integrada en la base.\n"
            "- Diseñada para oficinas en casa y sesiones de estudio nocturnas."
        ),
        "lámpara de escritorio lumenx": (
            "Producto: Lámpara de Escritorio LumenX\n"
            "- Brazo ajustable de tres puntos con rotación de 270°.\n"
            "- Espectro LED personalizado de cálido a neutro (2700K-4000K).\n"
            "- Almohadilla de carga USB-C integrada en la base.\n"
            "- Diseñada para oficinas en casa y sesiones de estudio nocturnas."
        )
    }
    return briefs.get(product_name.lower(), f"No hay resumen almacenado para '{product_name}'.")


# NOTA: approval_mode="never_require" es por brevedad del ejemplo.
@tool(approval_mode="never_require")
def get_brand_voice_profile(
    voice_name: Annotated[str, Field(description="Brand or campaign voice to emulate.")],
) -> str:
    """Return guidelines for the requested brand voice."""
    voices = {
        "lumenx launch": (
            "Directrices de voz:\n"
            "- Amigable y moderno con oraciones concisas.\n"
            "- Resaltar beneficios prácticos antes que estéticos.\n"
            "- Terminar con una invitación a imaginar el producto en uso diario."
        ),
        "lanzamiento lumenx": (
            "Directrices de voz:\n"
            "- Amigable y moderno con oraciones concisas.\n"
            "- Resaltar beneficios prácticos antes que estéticos.\n"
            "- Terminar con una invitación a imaginar el producto en uso diario."
        ),
    }
    return voices.get(voice_name.lower(), f"No hay perfil de voz almacenado para '{voice_name}'.")


@dataclass
class DraftFeedbackRequest:
    """Payload sent for human review."""

    prompt: str = ""
    draft_text: str = ""
    conversation: list[ChatMessage] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]


class Coordinator(Executor):
    """Bridge between the writer agent, human feedback, and the final editor."""

    def __init__(self, id: str, writer_id: str, final_editor_id: str) -> None:
        super().__init__(id)
        self.writer_id = writer_id
        self.final_editor_id = final_editor_id

    @handler
    async def on_writer_response(
        self,
        draft: AgentExecutorResponse,
        ctx: WorkflowContext[Never, AgentResponse],
    ) -> None:
        """Handle responses from the other two agents in the workflow."""
        if draft.executor_id == self.final_editor_id:
            # Respuesta del editor final; emitir salida directamente.
            await ctx.yield_output(draft.agent_response)
            return

        # Respuesta del agente escritor; solicitar retroalimentación humana.
        # Preservar la conversación completa para que el editor final
        # pueda ver los rastros de herramientas y el prompt inicial.
        conversation: list[ChatMessage]
        if draft.full_conversation is not None:
            conversation = list(draft.full_conversation)
        else:
            conversation = list(draft.agent_response.messages)
        draft_text = draft.agent_response.text.strip()
        if not draft_text:
            draft_text = "No se produjo ninguna versión preliminar."

        prompt = (
            "Revisa la versión preliminar del escritor y comparte una nota direccional breve "
            "(ajustes de tono, detalles imprescindibles, público objetivo, etc.). "
            "Mantén la nota en menos de 30 palabras."
        )
        await ctx.request_info(
            request_data=DraftFeedbackRequest(prompt=prompt, draft_text=draft_text, conversation=conversation),
            response_type=str,
        )

    @response_handler
    async def on_human_feedback(
        self,
        original_request: DraftFeedbackRequest,
        feedback: str,
        ctx: WorkflowContext[AgentExecutorRequest],
    ) -> None:
        note = feedback.strip()
        if note.lower() in {"approve", "aprobar"}:
            # El humano aprobó el borrador tal como está; reenviarlo sin cambios.
            await ctx.send_message(
                AgentExecutorRequest(
                    messages=original_request.conversation
                    + [ChatMessage(Role.USER, text="La versión preliminar está aprobada tal como está.")],
                    should_respond=True,
                ),
                target_id=self.final_editor_id,
            )
            return

        # El humano proporcionó retroalimentación; indicar al escritor que revise.
        conversation: list[ChatMessage] = list(original_request.conversation)
        instruction = (
            "Un revisor humano compartió la siguiente guía:\n"
            f"{note or 'No se proporcionó guía específica.'}\n\n"
            "Reescribe la versión preliminar del mensaje anterior del asistente en una versión final pulida. "
            "Mantén la respuesta en menos de 120 palabras y refleja los ajustes de tono solicitados."
        )
        conversation.append(ChatMessage(Role.USER, text=instruction))
        await ctx.send_message(
            AgentExecutorRequest(messages=conversation, should_respond=True), target_id=self.writer_id
        )


def create_writer_agent() -> ChatAgent:
    """Create a writer agent with tools."""
    return ChatAgent(
        chat_client=client,
        name="writer_agent",
        instructions=(
            "Eres un escritor de marketing. "
            "Llama a las herramientas disponibles antes de escribir una versión preliminar "
            "para ser preciso. "
            "Siempre llama a ambas herramientas una vez antes de escribir una versión preliminar. "
            "Resume las salidas de las herramientas como viñetas y luego produce una versión preliminar "
            "de 3 oraciones."
        ),
        tools=[fetch_product_brief, get_brand_voice_profile],
        tool_choice="required",
    )


def create_final_editor_agent() -> ChatAgent:
    """Create a final editor agent."""
    return ChatAgent(
        chat_client=client,
        name="final_editor_agent",
        instructions=(
            "Eres un editor que pule el texto de marketing después de la aprobación humana. "
            "Corrige cualquier problema legal o fáctico. Devuelve la versión final aunque no se necesiten cambios."
        ),
    )


def display_agent_run_update(event: AgentRunUpdateEvent, last_executor: str | None) -> None:
    """Display an AgentRunUpdateEvent in a readable format."""
    printed_tool_calls: set[str] = set()
    printed_tool_results: set[str] = set()
    executor_id = event.executor_id
    update = event.data
    # Extraer e imprimir cualquier nueva llamada a herramienta o resultado de la actualización.
    # Content.type indica el tipo de contenido: 'function_call', 'function_result', 'text', etc.
    function_calls = [c for c in update.contents if isinstance(c, Content) and c.type == "function_call"]
    function_results = [c for c in update.contents if isinstance(c, Content) and c.type == "function_result"]
    if executor_id != last_executor:
        if last_executor is not None:
            print()
        print(f"{executor_id}:", end=" ", flush=True)
        last_executor = executor_id
    # Imprimir cualquier nueva llamada a herramienta antes de la actualización de texto.
    for call in function_calls:
        if call.call_id in printed_tool_calls:
            continue
        printed_tool_calls.add(call.call_id)
        args = call.arguments
        args_preview = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else (args or "").strip()
        print(
            f"\n{executor_id} [llamada-herramienta] {call.name}({args_preview})",
            flush=True,
        )
        print(f"{executor_id}:", end=" ", flush=True)
    # Imprimir cualquier nuevo resultado de herramienta antes de la actualización de texto.
    for result in function_results:
        if result.call_id in printed_tool_results:
            continue
        printed_tool_results.add(result.call_id)
        result_text = result.result
        if not isinstance(result_text, str):
            result_text = json.dumps(result_text, ensure_ascii=False)
        print(
            f"\n{executor_id} [resultado-herramienta] {result.call_id}: {result_text}",
            flush=True,
        )
        print(f"{executor_id}:", end=" ", flush=True)
    # Finalmente, imprimir la actualización de texto.
    print(update, end="", flush=True)


async def main() -> None:
    """Run the workflow and connect human feedback between two agents."""

    # Construir el workflow.
    workflow = (
        WorkflowBuilder()
        .register_agent(create_writer_agent, name="writer_agent")
        .register_agent(create_final_editor_agent, name="final_editor_agent")
        .register_executor(
            lambda: Coordinator(
                id="coordinator",
                writer_id="writer_agent",
                final_editor_id="final_editor_agent",
            ),
            name="coordinator",
        )
        .set_start_executor("writer_agent")
        .add_edge("writer_agent", "coordinator")
        .add_edge("coordinator", "writer_agent")
        .add_edge("final_editor_agent", "coordinator")
        .add_edge("coordinator", "final_editor_agent")
        .build()
    )

    # Interruptor para activar la visualización de actualizaciones de ejecución del agente.
    # Por defecto está desactivado para reducir el desorden durante la entrada humana.
    display_agent_run_update_switch = False

    print(
        "Modo interactivo. Cuando se te solicite, proporciona una nota de retroalimentación breve para el editor.",
        flush=True,
    )

    pending_responses: dict[str, str] | None = None
    completed = False
    initial_run = True

    while not completed:
        last_executor: str | None = None
        if initial_run:
            stream = workflow.run_stream(
                "Crea un breve texto de lanzamiento para la lámpara de escritorio LumenX. "
                "Enfatiza la ajustabilidad y la iluminación cálida."
            )
            initial_run = False
        elif pending_responses is not None:
            stream = workflow.send_responses_streaming(pending_responses)
            pending_responses = None
        else:
            break

        requests: list[tuple[str, DraftFeedbackRequest]] = []

        async for event in stream:
            if isinstance(event, AgentRunUpdateEvent) and display_agent_run_update_switch:
                display_agent_run_update(event, last_executor)
            if isinstance(event, RequestInfoEvent) and isinstance(event.data, DraftFeedbackRequest):
                # Guardar la solicitud para solicitar al humano después de que se complete el stream.
                requests.append((event.request_id, event.data))
                last_executor = None
            elif isinstance(event, WorkflowOutputEvent):
                last_executor = None
                response = event.data
                print("\n===== Salida final =====")
                final_text = getattr(response, "text", str(response))
                print(final_text.strip())
                completed = True

        if requests and not completed:
            responses: dict[str, str] = {}
            for request_id, request in requests:
                print("\n----- Versión preliminar del escritor -----")
                print(request.draft_text.strip())
                print("\nProporciona guía para el editor (o 'approve'/'aprobar' para aceptar la versión preliminar).")
                answer = input("Retroalimentación humana: ").strip()  # noqa: ASYNC250
                if answer.lower() in {"exit", "salir"}:
                    print("Saliendo...")
                    return
                responses[request_id] = answer
            pending_responses = responses

    print("Workflow completado.")

    # Cerrar la credencial asíncrona si fue creada
    if async_credential is not None:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
