## Plan: Session 5 Workflow Builder Demos (File-by-File)

This plan is organized by proposed example file. For each file: (1) source examples/patterns to pull from, and (2) what the workflow should accomplish in the talk.

**Implementation order**
1. `workflow_fan_out_fan_in_edges.py` (template concatenation aggregator)
2. `workflow_aggregator_llm_summary.py` (Agent as aggregator — LLM synthesis)
3. `workflow_aggregator_structured.py` (Pydantic structured extraction)
4. `workflow_aggregator_voting.py` (ensemble classification — majority vote)
5. `workflow_aggregator_ranked.py` (generate N candidates — score & rank)
6. `workflow_rag_ingest_parallel.py` (parallelized version of `workflow_rag_ingest.py`)
7. `workflow_multi_selection_edge_group.py`
8. `workflow_agents_concurrent.py`
9. `workflow_concurrent_custom_aggregator.py`
10. `workflow_handoff_builder_rules.py`
11. `workflow_composition_as_agent.py`

## Proposed Files

### 1) `/workspace/examples/workflow_fan_out_fan_in_edges.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_converge.py` (branch-to-join narrative), `/workspace/examples/workflow_agents.py` (simple builder wiring)
  - Upstream: `python/samples/03-workflows/parallelism/fan_out_fan_in_edges.py`, `python/packages/core/tests/workflow/test_workflow.py`
  - Docs: Edges page (`add_fan_out_edges`, `add_fan_in_edges`)
- **Workflow accomplishes**
  - Shows explicit graph-level fan-out and fan-in APIs (not just orchestration abstraction).
  - Demonstrates true parallel branch execution and list-based aggregation at join.
  - Custom aggregator consolidates research, marketing, and legal perspectives into one report.

### 2) `/workspace/examples/workflow_aggregator_llm_summary.py`
- **Pull from**
  - Local: `workflow_fan_out_fan_in_edges.py` (same fan-out/fan-in wiring)
  - Docs: Edges page (`add_fan_out_edges`, `add_fan_in_edges`)
- **Workflow accomplishes**
  - Same 3 expert branches (researcher, marketer, legal) analyze a product launch.
  - Replaces the custom `Executor` aggregator with an **Agent** that synthesizes a concise executive brief.
  - Teaching point: the fan-in target can be an Agent — "let the LLM merge it." Shows the flexibility/cost tradeoff (natural output, but adds an LLM call + latency).

### 3) `/workspace/examples/workflow_aggregator_structured.py`
- **Pull from**
  - Local: `workflow_fan_out_fan_in_edges.py` (fan-out/fan-in wiring pattern)
  - Docs: Edges page
- **Workflow accomplishes**
  - Three interviewer agents (technical, behavioral, culture-fit) each assess a job candidate.
  - Aggregator Executor calls the LLM directly with `response_format=CandidateReview`, yielding a typed Pydantic model.
  - Teaching point: Executors can make their own LLM calls with structured output — aggregation can produce typed objects for downstream code, not just prose.

### 4) `/workspace/examples/workflow_aggregator_voting.py`
- **Pull from**
  - Local: `workflow_fan_out_fan_in_edges.py` (fan-out/fan-in wiring pattern)
  - Docs: Edges page
- **Workflow accomplishes**
  - Three classifier agents with **different reasoning strategies** (keyword-based, sentiment-based, intent-based) each categorize a support ticket.
  - Aggregator counts votes via `Counter` and picks the majority label.
  - Teaching point: fan-out gives redundancy/confidence for decisions; aggregation is pure logic. Comment notes that in production, different models per branch would strengthen the ensemble.

### 5) `/workspace/examples/workflow_aggregator_ranked.py`
- **Pull from**
  - Local: `workflow_fan_out_fan_in_edges.py` (fan-out/fan-in wiring pattern)
  - Docs: Edges page
- **Workflow accomplishes**
  - Three creative agents each propose a marketing slogan for a product.
  - Aggregator (Agent or Executor) scores each proposal on criteria (creativity, memorability, brand fit) and yields a ranked list.
  - Teaching point: aggregation can be evaluative — the fan-in step judges via LLM, not just collects. Pattern applies to "generate N candidates, pick the best."

### 6) `/workspace/examples/workflow_rag_ingest_parallel.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_rag_ingest.py` (extract/chunk/embed pipeline and provider bootstrap)
  - Upstream: `python/samples/03-workflows/parallelism/fan_out_fan_in_edges.py`, `python/samples/03-workflows/parallelism/map_reduce_and_visualization.py`
  - Docs: Edges and parallelism concepts (`add_fan_out_edges`, `add_fan_in_edges`)
- **Workflow accomplishes**
  - Parallelizes embedding generation by fan-out over chunk batches and fan-in aggregation of embedded chunks.
  - Keeps the same business goal as the current ingest demo (RAG ingestion) while changing the execution model.
  - Provides a direct comparison against `workflow_agents_concurrent.py` to show domain-level parallelism (specialist perspectives) vs data-level parallelism (batch processing).

### 7) `/workspace/examples/workflow_multi_selection_edge_group.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_switch_case.py` (edge-group style, structured routing)
  - Upstream: `python/samples/03-workflows/control-flow/multi_selection_edge_group.py`
  - Docs: Edges page “Multi-Selection Edge Group”
- **Workflow accomplishes**
  - Demonstrates one input selecting one-or-many downstream targets dynamically.
  - Shows subset fan-out behavior beyond switch-case’s single-target routing.
  - Covers the outline item for multi-selection edges with a deterministic branching example.

### 8) `/workspace/examples/workflow_agents_concurrent.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_agents_sequential.py` (provider/bootstrap style), `/workspace/examples/workflow_agents_streaming.py` (stream output handling)
  - Upstream: `python/samples/03-workflows/orchestrations/concurrent_agents.py`
  - Docs: Concurrent orchestration page (`ConcurrentBuilder().participants([...]).build()`)
- **Workflow accomplishes**
  - Runs 3 specialist agents in parallel on the same user prompt.
  - Demonstrates default concurrent aggregation (single combined message list output).
  - Gives the baseline “sequential vs concurrent” contrast for the talk.

### 9) `/workspace/examples/workflow_concurrent_custom_aggregator.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_magenticone.py` (summary-friendly output formatting), `/workspace/examples/workflow_agents_streaming.py`
  - Upstream: `python/samples/03-workflows/orchestrations/concurrent_custom_aggregator.py`
  - Docs: Concurrent “Advanced: Custom Aggregator” section
- **Workflow accomplishes**
  - Keeps concurrent fan-out but replaces default aggregation with `.with_aggregator(...)` callback.
  - Produces one concise synthesized answer suitable for live demo narration.
  - Teaches where custom post-processing belongs in builder-based orchestration.

### 10) `/workspace/examples/workflow_handoff_builder_rules.py`
- **Pull from**
  - Local: `/workspace/examples/agent_with_subagent.py` and `/workspace/examples/agent_supervisor.py` (architectural contrast language), plus common provider bootstrap from workflow examples
  - Upstream: `python/samples/03-workflows/orchestrations/handoff_simple.py`, `python/samples/03-workflows/orchestrations/README.md` handoff section
  - Docs: Handoff orchestration page (`with_start_agent`, `add_handoff`, `HandoffAgentUserRequest` loop)
- **Workflow accomplishes**
  - Creates a handoff workflow with explicit routing rules between triage and specialists.
  - Runs a scripted request/response loop (no live typing required) for deterministic demo.
  - Makes clear that handoff transfers control ownership, unlike supervisor/agent-as-tools patterns.

### 11) `/workspace/examples/workflow_composition_as_agent.py`
- **Pull from**
  - Local: `/workspace/examples/workflow_conditional_state_isolated.py` (factory/isolation mindset), `/workspace/examples/workflow_agents.py`
  - Upstream: `python/samples/03-workflows/agents/concurrent_workflow_as_agent.py`, `python/samples/03-workflows/composition/sub_workflow_basics.py`
  - Docs: Composition samples + orchestration `workflow.as_agent(...)` references
- **Workflow accomplishes**
  - Wraps a child workflow as an agent and composes it in a parent workflow.
  - Demonstrates composability/nesting for scaling complex systems.
  - Covers the “workflows can be nested or combined” talk objective directly.

## Reused Existing Files (No New File Needed)

### `/workspace/examples/workflow_agents_sequential.py`
- **Use in talk for**: sequential baseline before introducing concurrency.

### `/workspace/examples/workflow_magenticone.py`
- **Use in talk for**: planning supervisor (Magentic) orchestration pattern.

### `/workspace/examples/workflow_switch_case.py`
- **Use in talk for**: bridge from switch-case routing to multi-selection routing.

## Optional Appendix

### `/workspace/examples/workflow_asyncio_gather_vs_workflow.py`
- **When to use**: optional add-on only if time permits and the audience is comfortable with `asyncio` basics.
- **Purpose**: contrasts language-level concurrency (`asyncio.gather`) with framework-level orchestration.
- **Why appendix**: can distract from the core built-in workflow builder narrative for audiences new to async Python.

## Scope Boundaries
- Include only built-in workflow builder content for Session 5.
- Exclude HITL-focused demos (next session).
- Exclude Ignite Shop implementation in this repo (separate repo).
- Defer Spanish translations until English examples stabilize.

## Verification Plan
1. Smoke run each new script with standard provider env pattern used by current examples.
2. Verify each script’s terminal output demonstrates the intended teaching point above.
3. Run an explicit A/B comparison between `workflow_agents_concurrent.py` and `workflow_rag_ingest_parallel.py` and capture whether the audience sees a clear difference in orchestration pattern (specialist-agent parallelism vs data-pipeline parallelism).
4. Rehearse in talk order with one fallback per segment (if a demo fails, which demo replaces it).
## Open Questions for MAF Team
- **When is `output_executors` needed/recommended?** By default, `WorkflowBuilder` surfaces all outputs from all executors as events. If you only want outputs from certain executors, use `output_executors`. In our fan-out/fan-in demos, without `output_executors=[<aggregator>]`, the intermediate `Agent` nodes' `AgentResponse` objects leak into `get_outputs()` alongside the aggregator's output. We added `output_executors` to every fan-out/fan-in demo. Worth a slide callout — possibly on the same slide as the aggregation patterns table.