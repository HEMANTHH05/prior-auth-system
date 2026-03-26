"""
app/graph/graph.py

Assembles all nodes and routers into a compiled LangGraph application.

This is the single entry point for running the prior-authorization workflow.
FastAPI routes call graph_app.invoke() or graph_app.stream() with an initial
state dict; this file owns the wiring, checkpointing, and interrupt setup.

Graph topology (happy-path prior_auth, no attachments):

  [intent_classifier]
         │ route_after_intent
         ▼
  [member_context]
         │ route_after_member_context
         ▼
  [policy_retrieval]          ◄── document_ingestion feeds here if attachments exist
         │ fixed edge
         ▼
  [clinical_reasoning]
         │ fixed edge
         ▼
  [confidence_evaluation]
         │ route_after_confidence
         ├──(requires_human_review=True)──► [human_review] ──► [audit_output] ──► END
         └──(requires_human_review=False)──────────────────► [audit_output] ──► END

  Error / non-prior-auth shortcuts:
  intent_classifier ──► [error_output] ──► END
  member_context    ──► [error_output] ──► END    (validation_error)
  member_context    ──► [audit_output] ──► END    (eligibility / member_service)
"""

from __future__ import annotations

import logging
import os
import pprint
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    audit_output_node,
    clinical_reasoning_node,
    confidence_evaluation_node,
    document_ingestion_node,
    human_review_node,
    intent_classifier_node,
    member_context_node,
    policy_retrieval_node,
)
from app.graph.routers import (
    route_after_confidence,
    route_after_human_review,
    route_after_intent,
    route_after_member_context,
)
from app.graph.state import PriorAuthState, PriorAuthStateUpdate

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error output node
# ---------------------------------------------------------------------------


def error_output_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Terminal node for requests that cannot be processed.

    Handles two failure modes:
    1. Unrecognisable intent — the request type is not something this system
       can handle (e.g., a plain billing query sent to the wrong endpoint).
    2. Member not found — the member_id from the request has no matching row
       in the members table (invalid ID, test ID, etc.).

    In both cases we return a structured final_output so the FastAPI response
    layer can return a well-formed JSON error body rather than a 500.
    We do NOT write to the prior_auth_requests table — there is nothing
    meaningful to record for a request that never reached clinical processing.
    """
    validation_error: str | None = state.get("validation_error")
    intent: str = state.get("intent") or "unknown"

    if validation_error:
        message = validation_error
        error_type = "MEMBER_NOT_FOUND"
    elif intent == "unknown":
        message = (
            "The request could not be classified as a prior authorization, "
            "eligibility check, or member service inquiry. "
            "Please resubmit with a clear description of the requested service."
        )
        error_type = "UNKNOWN_INTENT"
    else:
        message = f"Request could not be processed (intent={intent!r})."
        error_type = "PROCESSING_ERROR"

    logger.warning(
        "[error_output] %s: %s", error_type, message
    )

    return {
        "final_output": {
            "status": "ERROR",
            "error_type": error_type,
            "message": message,
            "member_id": state.get("member_id"),
            "intent": intent,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct, wire, and compile the prior-authorization LangGraph.

    Why StateGraph(PriorAuthState):
    Passing our TypedDict schema gives LangGraph the field definitions it
    needs to validate state merges and generate the graph's input schema for
    LangServe / API documentation.

    Why MemorySaver (not PostgresSaver):
    MemorySaver stores checkpoints in an in-process Python dict.  It requires
    zero external infrastructure, making it ideal for local development and
    testing.  In production, swap it for AsyncPostgresSaver backed by Supabase
    by changing only the checkpointer= argument to compile() — no graph
    topology changes needed.

    Why interrupt_before=["human_review"]:
    LangGraph's interrupt mechanism pauses graph execution *before* the named
    node runs.  When the graph reaches the edge leading to human_review_node,
    it checkpoints the full state and suspends.  The workflow resumes only when
    graph_app.invoke() is called again with the same thread_id (supplied by the
    nurse review API endpoint).  This is the canonical LangGraph human-in-the-
    loop pattern — the node itself doesn't need any special suspend/resume code.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() / .stream().
    """
    workflow = StateGraph(PriorAuthState)

    # -----------------------------------------------------------------------
    # Register nodes
    # Node names here MUST match the strings returned by routers.py constants.
    # -----------------------------------------------------------------------

    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("member_context", member_context_node)
    workflow.add_node("document_ingestion", document_ingestion_node)
    workflow.add_node("policy_retrieval", policy_retrieval_node)
    workflow.add_node("clinical_reasoning", clinical_reasoning_node)
    workflow.add_node("confidence_evaluation", confidence_evaluation_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("audit_output", audit_output_node)
    workflow.add_node("error_output", error_output_node)

    # -----------------------------------------------------------------------
    # Entry point
    # Every graph invocation starts at intent_classifier regardless of input.
    # -----------------------------------------------------------------------

    workflow.set_entry_point("intent_classifier")

    # -----------------------------------------------------------------------
    # Edges
    # -----------------------------------------------------------------------

    # --- After intent_classifier ---
    # CONDITIONAL: intent/validation_error determines the first branch.
    # path_map explicitly lists every possible return value from the router so
    # LangGraph can validate the routing function at compile time and include
    # all reachable nodes in the graph visualisation.
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "member_context": "member_context",
            "error_output":   "error_output",
        },
    )

    # --- After member_context ---
    # CONDITIONAL: four possible destinations depending on intent and
    # attachment presence.  Non-prior-auth intents short-circuit to audit_output
    # (they only need member data).  Validation errors go to audit_output so
    # the API gets a structured error response (not error_output — the member
    # data fetch already logged the warning; audit assembles the response).
    workflow.add_conditional_edges(
        "member_context",
        route_after_member_context,
        {
            "document_ingestion": "document_ingestion",
            "policy_retrieval":   "policy_retrieval",
            "audit_output":       "audit_output",
            "error_output":       "error_output",
        },
    )

    # --- Fixed pipeline edges ---
    # document_ingestion always feeds into policy_retrieval.
    # The extracted procedure description improves the RAG query quality,
    # so document processing must complete before policy retrieval runs.
    workflow.add_edge("document_ingestion", "policy_retrieval")

    # policy_retrieval always feeds into clinical_reasoning.
    # Policy chunks must be loaded before the LLM prompt is assembled.
    workflow.add_edge("policy_retrieval", "clinical_reasoning")

    # clinical_reasoning always feeds into confidence_evaluation.
    # The LLM decision and confidence score must exist before the routing
    # logic that decides whether human review is needed.
    workflow.add_edge("clinical_reasoning", "confidence_evaluation")

    # --- After confidence_evaluation ---
    # CONDITIONAL: binary choice — human review required or not.
    workflow.add_conditional_edges(
        "confidence_evaluation",
        route_after_confidence,
        {
            "human_review": "human_review",
            "audit_output": "audit_output",
        },
    )

    # --- After human_review ---
    # CONDITIONAL: always routes to audit_output, but expressed as a
    # conditional edge (rather than a fixed edge) to leave a hook for future
    # routing logic (e.g., routing REQUEST_MORE_INFO to a waiting state).
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "audit_output": "audit_output",
        },
    )

    # --- Terminal edges ---
    # Both audit_output and error_output are terminal — nothing runs after them.
    workflow.add_edge("audit_output", END)
    workflow.add_edge("error_output", END)

    # -----------------------------------------------------------------------
    # Compile
    # -----------------------------------------------------------------------

    # MemorySaver: in-process dict-based checkpointer.
    # Thread safety note: MemorySaver is NOT thread-safe across concurrent
    # Python threads.  For production, replace with AsyncPostgresSaver.
    checkpointer = MemorySaver()

    compiled = workflow.compile(
        checkpointer=checkpointer,
        # interrupt_before pauses execution BEFORE human_review_node runs.
        # State is checkpointed at this point; the workflow resumes when
        # graph_app.invoke() is called again on the same thread_id with the
        # nurse's decision added to state.
        interrupt_before=["human_review"],
    )

    logger.info("Prior-auth LangGraph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Module-level compiled instance
# ---------------------------------------------------------------------------

# Compiled once at import time and reused across all FastAPI requests.
# LangGraph compiled graphs are stateless between invocations — all state
# is managed by the checkpointer, so sharing this instance is safe.
graph_app = build_graph()


# ---------------------------------------------------------------------------
# Test / smoke-test function
# ---------------------------------------------------------------------------


def test_graph() -> None:
    """Run the full graph with a real member from Supabase and print each step.

    Demonstrates:
    1. Full pipeline execution via graph_app.stream()
    2. The interrupt behaviour before human_review
    3. Workflow resume with a nurse decision

    Usage::

        source venv/bin/activate
        python -m app.graph.graph
    """
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)

    # thread_id ties this run to a specific LangGraph checkpoint thread.
    # Using a fixed value here so the test is repeatable.
    thread_id = "test-prior-auth-001"
    config = {"configurable": {"thread_id": thread_id}}

    # Real member from Supabase (confirmed in member_service smoke tests)
    initial_state: PriorAuthState = {
        "raw_request": (
            "Prior authorization request for MRI lumbar spine. "
            "Patient has 6 weeks of low back pain with radiculopathy. "
            "Failed conservative therapy including PT and NSAIDs."
        ),
        "member_id": "06e30f00-63d9-7a7d-d30b-79f32641f372",
        "cpt_code": "72148",
        "icd_code": "M54.5",
        "provider_npi": "1234567890",
        "clinical_notes": (
            "Patient presents with 6-week history of low back pain radiating "
            "to the left leg (L4-L5 distribution). Neurological symptoms include "
            "numbness and tingling in the left foot. Conservative treatment with "
            "6 weeks of physical therapy and NSAIDs (naproxen 500mg BID) has "
            "failed to provide adequate relief. Plain X-ray completed 2 weeks "
            "ago showed no fracture. Requesting MRI lumbar spine without "
            "contrast to evaluate for disc herniation or nerve root compression."
        ),
        "attachments": [],
        "is_urgent": False,
        "case_type": "outpatient",
        # Optional fields start as None / absent
        "intent": None,
        "member_context": None,
        "extracted_documents": None,
        "retrieved_policies": None,
        "reasoning_output": None,
        "confidence_score": None,
        "requires_human_review": None,
        "case_id": None,
        "nurse_decision": None,
        "nurse_id": None,
        "nurse_override_reason": None,
        "audit_written": None,
        "final_output": None,
        "validation_error": None,
        "thread_id": thread_id,
    }

    print("=" * 70)
    print("Prior Auth LangGraph — full workflow test")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Phase 1: stream until interrupt (before human_review if triggered)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Streaming graph until interrupt or completion...\n")

    last_node_output: dict[str, Any] = {}
    interrupted = False

    for step in graph_app.stream(initial_state, config=config):
        for node_name, node_output in step.items():
            print(f"  ── {node_name} ──")

            # Print selected fields rather than the full state dump
            _print_node_summary(node_name, node_output)
            last_node_output = node_output

    # Check if the graph paused at human_review
    snapshot = graph_app.get_state(config)
    if snapshot.next and "human_review" in snapshot.next:
        interrupted = True
        print("\n[!] Graph interrupted before human_review_node.")
        print("    Workflow is waiting for nurse decision.")
        print(f"    Resume with thread_id={thread_id!r}")

        # -----------------------------------------------------------------------
        # Phase 2: simulate nurse review and resume
        # -----------------------------------------------------------------------
        print("\n[Phase 2] Simulating nurse APPROVE decision and resuming...\n")

        # In production, the nurse submits via the API which calls:
        #   graph_app.invoke(nurse_update, config=config)
        # Here we simulate that resume call directly.
        nurse_update: dict[str, Any] = {
            "nurse_decision": "APPROVED",
            "nurse_id": "RN-TEST-001",
            "nurse_override_reason": None,
        }

        for step in graph_app.stream(nurse_update, config=config):
            for node_name, node_output in step.items():
                print(f"  ── {node_name} ──")
                _print_node_summary(node_name, node_output)

    # -----------------------------------------------------------------------
    # Final state
    # -----------------------------------------------------------------------
    final_snapshot = graph_app.get_state(config)
    final_output = final_snapshot.values.get("final_output") or {}

    print("\n" + "=" * 70)
    print("FINAL OUTPUT:")
    print("=" * 70)
    for key, value in final_output.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        elif isinstance(value, str) and len(value) > 120:
            print(f"  {key}: {value[:120]}...")
        else:
            print(f"  {key}: {value}")


def _print_node_summary(node_name: str, output: dict[str, Any]) -> None:
    """Print a concise per-node summary for the test output."""
    summaries: dict[str, list[str]] = {
        "intent_classifier": ["intent"],
        "member_context": ["validation_error"],
        "document_ingestion": ["extraction_success", "extraction_warning"],
        "policy_retrieval": [],
        "clinical_reasoning": ["decision", "confidence_score"],
        "confidence_evaluation": ["requires_human_review"],
        "human_review": [],
        "audit_output": ["case_id", "audit_written"],
        "error_output": [],
    }

    keys_to_show = summaries.get(node_name, [])

    # For dict-valued state fields, drill into them for the summary keys
    flat: dict[str, Any] = {}
    for k, v in output.items():
        flat[k] = v
        if isinstance(v, dict):
            for nested_k, nested_v in v.items():
                flat[nested_k] = nested_v

    for key in keys_to_show:
        val = flat.get(key, "(not set)")
        print(f"    {key}: {val}")

    # Special summaries
    if node_name == "member_context":
        ctx = output.get("member_context") or {}
        if ctx:
            basic = ctx.get("basic_info") or {}
            name = f"{basic.get('first_name', '')} {basic.get('last_name', '')}".strip()
            print(f"    member_name: {name!r}")
            print(f"    conditions: {len(ctx.get('conditions') or [])}")
            print(f"    insurance_plans: {len(ctx.get('insurance') or [])}")

    if node_name == "policy_retrieval":
        policies = output.get("retrieved_policies") or []
        print(f"    chunks_retrieved: {len(policies)}")
        if policies:
            print(f"    top_chunk_score: {policies[0].get('score', 'N/A'):.4f}")
            print(f"    top_chunk_file:  {policies[0].get('filename', 'N/A')}")

    if node_name == "clinical_reasoning":
        ro = output.get("reasoning_output") or {}
        print(f"    decision: {ro.get('decision', '(not set)')}")
        print(f"    confidence_score: {output.get('confidence_score', '(not set)')}")

    if node_name in ("audit_output", "error_output"):
        fo = output.get("final_output") or {}
        print(f"    status: {fo.get('status', '(not set)')}")
        if node_name == "audit_output":
            print(f"    case_id: {fo.get('case_id', '(not set)')}")
            print(f"    audit_written: {fo.get('audit_written', '(not set)')}")


if __name__ == "__main__":
    test_graph()
