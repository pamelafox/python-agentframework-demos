"""Deterministic multi-selection routing with one-or-many targets.

Demonstrates: WorkflowBuilder.add_multi_selection_edge_group where a single
source message can activate one or multiple downstream executors.

Run:
    uv run examples/workflow_multi_selection_edge_group.py
    uv run examples/workflow_multi_selection_edge_group.py --devui  (opens DevUI at http://localhost:8099)
"""

import asyncio
import sys
from dataclasses import dataclass

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from typing_extensions import Never


@dataclass
class Ticket:
    """Structured representation of a support ticket."""

    text: str
    is_bug: bool
    is_billing: bool
    is_urgent: bool


class ParseTicketExecutor(Executor):
    """Parse incoming text into typed routing metadata."""

    @handler
    async def parse(self, text: str, ctx: WorkflowContext[Ticket]) -> None:
        """Convert raw text to a Ticket with routing flags."""
        lower = text.lower()
        ticket = Ticket(
            text=text,
            is_bug=("bug" in lower or "error" in lower or "crash" in lower),
            is_billing=("billing" in lower or "invoice" in lower or "charge" in lower),
            is_urgent=("urgent" in lower or "asap" in lower),
        )
        await ctx.send_message(ticket)


class SupportExecutor(Executor):
    """Default customer support handler."""

    @handler
    async def handle(self, ticket: Ticket, ctx: WorkflowContext[Never, str]) -> None:
        """Emit support handling output."""
        urgency = "high" if ticket.is_urgent else "normal"
        await ctx.yield_output(f"[Support] Opened {urgency} priority support case for: {ticket.text}")


class EngineeringExecutor(Executor):
    """Engineering triage handler for bug-related tickets."""

    @handler
    async def handle(self, ticket: Ticket, ctx: WorkflowContext[Never, str]) -> None:
        """Emit engineering handling output."""
        await ctx.yield_output(f"[Engineering] Routed bug triage: {ticket.text}")


class BillingExecutor(Executor):
    """Billing operations handler for charge/invoice issues."""

    @handler
    async def handle(self, ticket: Ticket, ctx: WorkflowContext[Never, str]) -> None:
        """Emit billing handling output."""
        await ctx.yield_output(f"[Billing] Routed billing review: {ticket.text}")


def select_targets(ticket: Ticket, target_ids: list[str]) -> list[str]:
    """Select one or many downstream targets based on ticket metadata.

    Expected order for ``target_ids``:
    [support_id, engineering_id, billing_id]
    """
    support_id, engineering_id, billing_id = target_ids

    selected = [support_id]
    if ticket.is_bug:
        selected.append(engineering_id)
    if ticket.is_billing:
        selected.append(billing_id)
    return selected


parse_ticket = ParseTicketExecutor(id="parse_ticket")
support = SupportExecutor(id="support")
engineering = EngineeringExecutor(id="engineering")
billing = BillingExecutor(id="billing")

workflow = (
    WorkflowBuilder(
        name="MultiSelectionEdgeGroup",
        description="One input can route to one-or-many targets via a selection function.",
        start_executor=parse_ticket,
    )
    .add_multi_selection_edge_group(
        parse_ticket,
        [support, engineering, billing],
        selection_func=select_targets,
    )
    .build()
)


async def main() -> None:
    """Run three deterministic routing examples."""
    samples = [
        "Urgent: app crashes on login with error 500.",
        "Question about billing charge on my invoice.",
        "Feature request: add dark mode.",
    ]

    for sample in samples:
        print(f"\nTicket: {sample}")
        events = await workflow.run(sample)
        for output in events.get_outputs():
            print(f"  {output}")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8099, auto_open=True)
    else:
        asyncio.run(main())
