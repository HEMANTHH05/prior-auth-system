"""
app/graph/state.py

LangGraph state schema for the Prior Authorization AI workflow.

Every node in the graph receives the full state dict and returns a partial
dict (PriorAuthStateUpdate) containing only the fields it modifies.
LangGraph merges the returned partial dict into the running state before
passing it to the next node.

Workflow node order:
  1. intent_classifier_node    — classifies request type
  2. member_context_node       — fetches member data from Supabase
  3. document_ingestion_node   — extracts text from PDF attachments
  4. policy_retrieval_node     — retrieves relevant policy chunks from Azure AI Search
  5. clinical_reasoning_node   — LLM makes approve/deny recommendation
  6. confidence_evaluation_node — decides whether human review is needed
  7. human_review_node         — nurse approve/deny (conditional, interrupt-based)
  8. audit_output_node         — writes final record to DB and assembles response

State is persisted between nodes via LangGraph's checkpointer (PostgreSQL-backed
SqliteSaver or AsyncPostgresSaver).  The thread_id field ties state to a
specific LangGraph thread so cases can be resumed after human review interrupts.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class PriorAuthState(TypedDict):
    """Complete state schema for the prior authorization LangGraph workflow.

    Divided into three logical groups:
      - Entry fields:   provided at graph.invoke() time; never mutated by nodes
      - Node fields:    set exactly once by the node listed in each comment
      - Routing fields: inspected by conditional edges to decide next node
    """

    # =========================================================================
    # ENTRY FIELDS
    # Set once at graph.invoke() and read by multiple downstream nodes.
    # Treated as immutable throughout the workflow.
    # =========================================================================

    raw_request: str
    """The original prior authorization request text as received from the API.
    Stored verbatim for audit trail completeness — the final audit record
    must include the source request exactly as submitted."""

    member_id: str
    """UUID of the health-plan member from the members table.
    Used by member_context_node (node 2) as the primary key for all Supabase
    queries; echoed in the final response and audit record."""

    cpt_code: str
    """5-digit Current Procedural Terminology code for the requested service.
    Used by policy_retrieval_node (node 4) to filter the Azure AI Search index
    and by clinical_reasoning_node (node 5) as the core subject of the
    medical-necessity evaluation."""

    icd_code: str
    """ICD-10-CM diagnosis code justifying the procedure.
    Combined with cpt_code to form the RAG query in policy_retrieval_node
    and included verbatim in the LLM prompt for clinical_reasoning_node."""

    provider_npi: str
    """10-digit National Provider Identifier of the requesting physician.
    Written to the prior_auth_requests table by audit_output_node (node 8)
    and used to verify network status against member insurance records."""

    clinical_notes: str
    """Free-text clinical documentation provided by the physician at submission.
    Primary input to the LLM in clinical_reasoning_node (node 5); combined
    with extracted_documents text to form the full clinical evidence package."""

    attachments: list[str]
    """List of file paths pointing to PDF attachments uploaded by the physician.
    Consumed by document_ingestion_node (node 3) which reads each path,
    extracts text, and stores the result in extracted_documents."""

    is_urgent: bool
    """True when the provider has flagged this as an urgent/expedited request.
    Read by confidence_evaluation_node (node 6) to lower the auto-approve
    threshold — urgent cases are more likely to be routed to human review.
    Also affects the estimated_decision_time in the final response."""

    case_type: str
    """Clinical category of the request: inpatient | outpatient | medication |
    dme | behavioral.  Used by policy_retrieval_node (node 4) to select the
    correct Azure AI Search filter scope."""

    # =========================================================================
    # NODE-POPULATED FIELDS
    # Each field is Optional because it does not exist until its owning node
    # runs.  Nodes that read a field must be downstream of the node that sets it.
    # =========================================================================

    intent: Optional[str]
    """Set by: intent_classifier_node (node 1).
    Classified intent of the incoming request.
    Values: "prior_auth" | "eligibility_check" | "member_service" | "unknown".
    Used by the first conditional edge to route non-prior-auth requests away
    from the clinical workflow immediately."""

    member_context: Optional[dict]
    """Set by: member_context_node (node 2).
    Complete member data package from Supabase, structured as a flat dict with
    keys: basic_info, insurance, claims, conditions, medications, imaging,
    procedures, prior_auth_history, validation_error.
    Injected into the LLM prompt in clinical_reasoning_node and written to the
    audit record by audit_output_node."""

    extracted_documents: Optional[dict]
    """Set by: document_ingestion_node (node 3).
    Structured extraction result from parse_clinical_document(), containing:
    patient_name, diagnosis, procedure_requested, conservative_treatment,
    treatment_duration, prior_imaging, medical_necessity, raw_text, and
    extraction metadata (extraction_success, extraction_warning).
    Appended to clinical_notes when building the LLM prompt in node 5."""

    retrieved_policies: Optional[list[dict]]
    """Set by: policy_retrieval_node (node 4).
    List of policy document chunks from Azure AI Search, each dict containing:
    id, content, filename, plan_name, procedure_type, score.
    Injected into the LLM system prompt as the authoritative policy grounding
    for clinical_reasoning_node so the LLM cites specific coverage criteria."""

    reasoning_output: Optional[dict]
    """Set by: clinical_reasoning_node (node 5).
    Validated ClinicalReasoningOutput dict with keys:
    decision (APPROVE|DENY|INSUFFICIENT_INFO), confidence_score, policy_basis,
    reasoning, missing_information, criteria_met, criteria_not_met.
    Read by confidence_evaluation_node (node 6) and written to the DB by
    audit_output_node (node 8)."""

    confidence_score: Optional[float]
    """Set by: clinical_reasoning_node (node 5).
    Extracted from reasoning_output for convenient access by downstream nodes
    without unpacking the full reasoning dict.
    Value in [0.0, 1.0]; below CONFIDENCE_THRESHOLD triggers human review."""

    requires_human_review: Optional[bool]
    """Set by: confidence_evaluation_node (node 6).
    True when the case must be routed to human_review_node (node 7) before
    a final decision is issued.  Evaluated from: confidence_score below
    threshold, INSUFFICIENT_INFO decision, is_urgent flag, or deny decision
    requiring nurse confirmation per policy."""

    case_id: Optional[str]
    """Set by: audit_output_node (node 8).
    UUID generated and assigned to this prior-auth case when the final record
    is written to the prior_auth_requests table.
    Returned in the API response so the provider can poll case status."""

    nurse_decision: Optional[str]
    """Set by: human_review_node (node 7) after nurse submits review.
    Values: "APPROVED" | "DENIED" | "REQUEST_MORE_INFO".
    Supersedes reasoning_output.decision as the authoritative final disposition
    when human review occurs; written to DB by audit_output_node."""

    nurse_id: Optional[str]
    """Set by: human_review_node (node 7).
    Employee/credential ID of the reviewing nurse (e.g., "RN-78901").
    Required for HIPAA audit trail; every human decision must be attributable
    to a specific licensed clinician in the final audit record."""

    nurse_override_reason: Optional[str]
    """Set by: human_review_node (node 7).
    Populated when the nurse's decision differs from the AI recommendation.
    Documents the clinical or policy rationale for the override; included
    verbatim in the audit record for appeals and regulatory review."""

    audit_written: Optional[bool]
    """Set by: audit_output_node (node 8).
    True once the complete case record has been successfully written to the
    prior_auth_requests table.  Checked by the graph's end condition to
    confirm the workflow completed without a silent DB failure."""

    final_output: Optional[dict]
    """Set by: audit_output_node (node 8).
    Complete case summary dict returned to the FastAPI response layer, with
    keys: case_id, status, member_id, ai_recommendation, confidence_score,
    requires_human_review, nurse_decision, submitted_at, message,
    estimated_decision_time.  Maps directly to CaseStatusResponse schema."""

    # =========================================================================
    # ROUTING FIELDS
    # Read by conditional edge functions to decide which node runs next.
    # =========================================================================

    validation_error: Optional[str]
    """Set by: member_context_node (node 2) when member_id is not found, or
    by intent_classifier_node (node 1) when input fails basic validation.
    Non-None value causes the conditional edge after node 2 to short-circuit
    to audit_output_node directly, skipping all clinical processing nodes.
    The error message is included in the final_output for the API response."""

    thread_id: Optional[str]
    """LangGraph thread identifier assigned at graph.invoke() time.
    Required by the checkpointer to persist and resume state across the
    human_review_node interrupt — when execution pauses for nurse input,
    the thread_id is what allows graph.invoke() to be called again on the
    same in-progress case without restarting from the beginning."""


# =============================================================================
# Type aliases
# =============================================================================

PriorAuthStateUpdate = dict
"""Return type for all LangGraph node functions.

Every node returns a plain dict containing only the state fields it modifies.
LangGraph merges this partial update into the running PriorAuthState.

Usage in node functions::

    def my_node(state: PriorAuthState) -> PriorAuthStateUpdate:
        ...
        return {"field_name": value}

Using a plain dict (rather than PriorAuthState) as the return type is the
LangGraph convention — nodes must not return the full state, only their delta.
"""
