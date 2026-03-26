"""
app/graph/routers.py

Conditional edge routing functions for the Prior Authorization AI workflow.

Each function is passed to StateGraph.add_conditional_edges() and receives the
full state snapshot at that point in the graph.  It returns a string that must
exactly match a node name (or the special END sentinel) registered in graph.py.

Routing overview:
                         ┌─────────────────────────────┐
  [intent_classifier] ──►│  route_after_intent         │──► member_context
                         │                             │──► error_output
                         └─────────────────────────────┘

                         ┌─────────────────────────────┐
  [member_context]    ──►│  route_after_member_context │──► document_ingestion
                         │                             │──► policy_retrieval
                         │                             │──► audit_output
                         │                             │──► error_output
                         └─────────────────────────────┘

                         ┌─────────────────────────────┐
  [confidence_eval]   ──►│  route_after_confidence     │──► human_review
                         │                             │──► audit_output
                         └─────────────────────────────┘

                         ┌─────────────────────────────┐
  [human_review]      ──►│  route_after_human_review   │──► audit_output
                         └─────────────────────────────┘

Node name constants below MUST stay in sync with the node names registered in
graph.py via StateGraph.add_node().
"""

from __future__ import annotations

import logging

from app.graph.state import PriorAuthState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# These match the strings passed to StateGraph.add_node() in graph.py exactly.
# Centralising them here means a rename in graph.py only needs one change here.
# ---------------------------------------------------------------------------

_NODE_MEMBER_CONTEXT = "member_context"
_NODE_DOCUMENT_INGESTION = "document_ingestion"
_NODE_POLICY_RETRIEVAL = "policy_retrieval"
_NODE_HUMAN_REVIEW = "human_review"
_NODE_AUDIT_OUTPUT = "audit_output"
_NODE_ERROR_OUTPUT = "error_output"


# ---------------------------------------------------------------------------
# ROUTER 1: route_after_intent
# ---------------------------------------------------------------------------


def route_after_intent(state: PriorAuthState) -> str:
    """Route after intent_classifier_node (node 1).

    Why we always go to member_context for valid intents:
    Regardless of whether this is a prior_auth, eligibility_check, or
    member_service request, the next logical step is always to load the
    member's data from Supabase.  Eligibility checks and member service
    requests need basic_info and insurance data; prior_auth requests need
    the full clinical context.  Centralising member data loading in node 2
    avoids each downstream branch duplicating that fetch.

    The only reason to skip member_context is if we already know the request
    is invalid (validation_error set during intent classification, e.g. missing
    required fields) or if the intent is unrecognisable.

    Reads:  intent, validation_error
    Returns: node name string
    """
    intent: str = state.get("intent") or "unknown"
    validation_error: str | None = state.get("validation_error")

    # Short-circuit to error output immediately if something went wrong
    # during intent classification (e.g. LLM call failed and left a
    # validation_error set by an upstream guardrail).
    if validation_error:
        logger.debug(
            "[router1] route_after_intent → %s (validation_error: %s)",
            _NODE_ERROR_OUTPUT, validation_error,
        )
        return _NODE_ERROR_OUTPUT

    # Unknown intent means we cannot safely process the request — return
    # an error rather than guessing which workflow path to follow.
    if intent == "unknown":
        logger.debug(
            "[router1] route_after_intent → %s (intent='unknown')",
            _NODE_ERROR_OUTPUT,
        )
        return _NODE_ERROR_OUTPUT

    # All recognised intents (prior_auth, eligibility_check, member_service)
    # proceed to member_context.  The finer routing based on intent happens
    # in route_after_member_context once we have the member data loaded.
    logger.debug(
        "[router1] route_after_intent → %s (intent=%r)",
        _NODE_MEMBER_CONTEXT, intent,
    )
    return _NODE_MEMBER_CONTEXT


# ---------------------------------------------------------------------------
# ROUTER 2: route_after_member_context
# ---------------------------------------------------------------------------


def route_after_member_context(state: PriorAuthState) -> str:
    """Route after member_context_node (node 2).

    Three distinct paths diverge here based on intent and available attachments:

    1. Error path (validation_error set):
       The member was not found in Supabase.  There is no point running any
       clinical processing — short-circuit directly to audit_output which will
       produce an error response for the API caller.

    2. Non-prior-auth intents (eligibility_check, member_service):
       These requests only need the member data that was just loaded.
       Routing them directly to audit_output skips all clinical processing
       nodes (document ingestion, RAG, LLM reasoning) which would add latency
       and cost for zero benefit.

    3. Prior-auth with attachments → document_ingestion:
       PDF attachments must be processed before policy retrieval so that the
       extracted procedure description improves RAG query quality in node 4.

    4. Prior-auth without attachments → policy_retrieval:
       Skip document_ingestion entirely — it would return an empty result
       anyway, and bypassing it saves one unnecessary node invocation.

    Reads:  validation_error, intent, attachments
    Returns: node name string
    """
    validation_error: str | None = state.get("validation_error")
    intent: str = state.get("intent") or "unknown"
    attachments: list = state.get("attachments") or []

    # Member not found — no clinical processing is meaningful without a member.
    if validation_error:
        logger.debug(
            "[router2] route_after_member_context → %s (validation_error: %s)",
            _NODE_AUDIT_OUTPUT, validation_error,
        )
        return _NODE_AUDIT_OUTPUT

    # Non-prior-auth requests are fully resolved once we have member context.
    if intent in ("eligibility_check", "member_service"):
        logger.debug(
            "[router2] route_after_member_context → %s (intent=%r, no clinical processing needed)",
            _NODE_AUDIT_OUTPUT, intent,
        )
        return _NODE_AUDIT_OUTPUT

    # Prior-auth: decide whether to process PDFs first.
    if attachments:
        logger.debug(
            "[router2] route_after_member_context → %s (%d attachment(s) to process)",
            _NODE_DOCUMENT_INGESTION, len(attachments),
        )
        return _NODE_DOCUMENT_INGESTION

    # Prior-auth with no attachments — jump straight to policy retrieval.
    logger.debug(
        "[router2] route_after_member_context → %s (no attachments, skipping document ingestion)",
        _NODE_POLICY_RETRIEVAL,
    )
    return _NODE_POLICY_RETRIEVAL


# ---------------------------------------------------------------------------
# ROUTER 3: route_after_confidence
# ---------------------------------------------------------------------------


def route_after_confidence(state: PriorAuthState) -> str:
    """Route after confidence_evaluation_node (node 6).

    Why this is a binary choice:
    The confidence evaluation node has already applied all the business rules
    (confidence threshold, DENY confirmation requirement, urgent flag, etc.)
    and distilled them into a single boolean.  The router simply reads that
    flag — it does not re-evaluate any rules itself.  This keeps routing logic
    thin and business logic in the node where it can be tested in isolation.

    human_review path:
    The workflow pauses at human_review_node via LangGraph's interrupt
    mechanism.  Execution resumes only after a nurse submits a decision via
    the API, which calls graph.invoke() again on the same thread_id.

    audit_output path:
    High-confidence APPROVE cases with no other review triggers can be
    written to the DB and returned to the caller immediately without waiting
    for a human.

    Reads:  requires_human_review
    Returns: node name string
    """
    requires_human_review: bool = bool(state.get("requires_human_review"))

    if requires_human_review:
        logger.debug(
            "[router3] route_after_confidence → %s (requires_human_review=True)",
            _NODE_HUMAN_REVIEW,
        )
        return _NODE_HUMAN_REVIEW

    logger.debug(
        "[router3] route_after_confidence → %s (requires_human_review=False)",
        _NODE_AUDIT_OUTPUT,
    )
    return _NODE_AUDIT_OUTPUT


# ---------------------------------------------------------------------------
# ROUTER 4: route_after_human_review
# ---------------------------------------------------------------------------


def route_after_human_review(state: PriorAuthState) -> str:
    """Route after human_review_node (node 7).

    Why this always returns audit_output:
    Once a nurse has recorded a decision (APPROVED / DENIED / REQUEST_MORE_INFO)
    the only remaining step is to write the authoritative case record to the
    database and assemble the final API response.  There are no branching
    conditions after human review — the nurse's decision is always final and
    always gets persisted.

    This router exists as an explicit function (rather than a direct edge) so
    that graph.py remains consistent — all nodes after which the path could
    theoretically diverge use a router function.  It also makes the graph
    diagram easier to read and leaves a hook for future routing logic (e.g.,
    routing REQUEST_MORE_INFO cases back to a waiting state).

    Reads:  nurse_decision (logged only — not used for routing)
    Returns: node name string (always audit_output)
    """
    nurse_decision: str | None = state.get("nurse_decision")

    logger.debug(
        "[router4] route_after_human_review → %s (nurse_decision=%r)",
        _NODE_AUDIT_OUTPUT, nurse_decision,
    )
    return _NODE_AUDIT_OUTPUT
