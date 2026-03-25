"""
app/graph/nodes.py

All 8 LangGraph node functions for the Prior Authorization AI workflow.

Each node:
  - Accepts a PriorAuthState dict (full state snapshot)
  - Returns a PriorAuthStateUpdate dict (only the fields it modifies)
  - Never raises — on failure it returns safe error state
  - Logs entry and exit at INFO level

Node execution order in the happy path:
  1. intent_classifier_node
  2. member_context_node
  3. document_ingestion_node
  4. policy_retrieval_node
  5. clinical_reasoning_node
  6. confidence_evaluation_node
  7. human_review_node          (conditional — only when requires_human_review)
  8. audit_output_node

The graph.py file wires these nodes together with edges and conditional routing.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.graph.state import PriorAuthState, PriorAuthStateUpdate
from app.models.schemas import AIDecision, ClinicalReasoningOutput
from app.services.doc_service import process_attachment
from app.services.member_service import get_full_member_context
from app.services.rag_service import search_policies

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

# Model name is configurable via .env so we can swap gpt-4o-mini ↔ gpt-4o
# for production without changing code.
_MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Cases with confidence below this threshold are always routed to human review.
# Read from env so ops can tune the threshold without a code deployment.
_CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))

_DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

# Lazy-initialised OpenAI client — created once per process, reused across nodes.
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client


# ---------------------------------------------------------------------------
# NODE 1: intent_classifier_node
# ---------------------------------------------------------------------------


def intent_classifier_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Classify the incoming request type using the LLM.

    Why this node exists:
    The same API endpoint may receive prior-auth requests, eligibility queries,
    or general member service inquiries.  Classifying intent first lets the
    conditional edge after this node immediately route non-prior-auth requests
    to a lightweight response path without firing the full clinical pipeline.

    Reads:  raw_request, cpt_code, icd_code
    Sets:   intent
    Routes: graph edge checks intent == "prior_auth" to continue workflow
    """
    logger.info("[node1] intent_classifier_node starting")

    # Build a minimal context string — we don't need clinical notes here,
    # just enough signal for the classifier to distinguish request types.
    context = (
        f"Request text: {state.get('raw_request', '')}\n"
        f"CPT code present: {'yes' if state.get('cpt_code') else 'no'}\n"
        f"ICD code present: {'yes' if state.get('icd_code') else 'no'}"
    )

    system_prompt = """You are a healthcare request classifier.
Classify the incoming request into exactly one of these intent categories:
- prior_auth: A request for prior authorization of a medical procedure or service
- eligibility_check: A request to verify member eligibility or coverage
- member_service: A general member service inquiry (billing, ID card, etc.)
- unknown: Cannot be classified into the above categories

Respond with a single JSON object: {"intent": "<category>"}
No explanation, no additional fields."""

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0,                           # deterministic classification
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        intent = parsed.get("intent", "unknown")

        # Validate the returned intent is one of our known values
        valid_intents = {"prior_auth", "eligibility_check", "member_service", "unknown"}
        if intent not in valid_intents:
            logger.warning("[node1] LLM returned unexpected intent '%s'; defaulting to 'unknown'", intent)
            intent = "unknown"

    except Exception as exc:
        # If classification fails, default to prior_auth so the workflow
        # continues — a false positive on routing is safer than dropping a
        # legitimate prior-auth request silently.
        logger.error("[node1] Intent classification failed: %s — defaulting to 'prior_auth'", exc)
        intent = "prior_auth"

    logger.info("[node1] intent_classifier_node complete | intent=%s", intent)
    return {"intent": intent}


# ---------------------------------------------------------------------------
# NODE 2: member_context_node
# ---------------------------------------------------------------------------


def member_context_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Fetch full member data from Supabase and load it into state.

    Why this node exists:
    All downstream nodes (policy retrieval, clinical reasoning, audit) need
    member demographics, insurance plan, conditions, medications, and prior
    auth history.  Centralising the Supabase fetch here means every other node
    can read from state without making its own DB calls.

    Reads:  member_id
    Sets:   member_context, validation_error
    Routes: validation_error triggers short-circuit to audit_output_node
    """
    member_id: str = state.get("member_id", "")
    logger.info("[node2] member_context_node starting | member_id=%s", member_id)

    result = get_full_member_context(member_id)

    if result.get("validation_error"):
        logger.warning(
            "[node2] Validation error for member_id=%s: %s",
            member_id,
            result["validation_error"],
        )
        return {
            "member_context": None,
            "validation_error": result["validation_error"],
        }

    logger.info(
        "[node2] member_context_node complete | "
        "conditions=%d, medications=%d, insurance=%d",
        len(result.get("conditions", [])),
        len(result.get("medications", [])),
        len(result.get("insurance", [])),
    )
    return {
        "member_context": result,
        "validation_error": None,
    }


# ---------------------------------------------------------------------------
# NODE 3: document_ingestion_node
# ---------------------------------------------------------------------------


def document_ingestion_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Extract and parse text from clinical PDF attachments.

    Why this node exists:
    Physicians often provide supporting clinical evidence as PDF attachments
    (imaging reports, H&P notes, lab results).  Extracting that text before
    the clinical reasoning node ensures the LLM has the complete clinical
    picture, not just the free-text clinical_notes field.

    Combination strategy for multiple attachments:
    Raw text from all PDFs is concatenated.  Structured list fields
    (conservative_treatment, prior_imaging) are merged and deduplicated.
    Scalar fields (diagnosis, procedure_requested, patient_name) are taken
    from the first attachment that successfully parses them — typically the
    primary physician narrative letter.

    Reads:  attachments
    Sets:   extracted_documents
    """
    attachments: list[str] = state.get("attachments", [])
    logger.info(
        "[node3] document_ingestion_node starting | %d attachment(s)", len(attachments)
    )

    # --- No attachments: return a safe empty result ---
    # This is not an error — many valid prior-auth requests have no PDFs.
    # We still populate extracted_documents so node 5 doesn't need to check
    # for its existence before reading keys.
    if not attachments:
        logger.info("[node3] No attachments provided; skipping extraction.")
        return {
            "extracted_documents": {
                "extraction_success": False,
                "extraction_warning": "No attachments provided",
                "raw_text": "",
                "patient_name": None,
                "physician_name": None,
                "diagnosis": None,
                "procedure_requested": None,
                "conservative_treatment": [],
                "treatment_duration": None,
                "prior_imaging": [],
                "medical_necessity": None,
            }
        }

    # --- Process each attachment ---
    all_results: list[dict[str, Any]] = []
    for path in attachments:
        try:
            result = process_attachment(path)
            all_results.append(result)
            logger.info(
                "[node3] Processed attachment '%s' | success=%s",
                path,
                result.get("extraction_success"),
            )
        except Exception as exc:
            logger.error("[node3] Failed to process attachment '%s': %s", path, exc)
            all_results.append({
                "extraction_success": False,
                "extraction_warning": str(exc),
                "raw_text": "",
                "conservative_treatment": [],
                "prior_imaging": [],
            })

    # --- Combine multiple attachment results into one dict ---
    # Concatenate all successfully-extracted raw text so the LLM has the
    # full document corpus in a single string.
    combined_raw_text = "\n\n---\n\n".join(
        r.get("raw_text", "") for r in all_results if r.get("raw_text")
    )

    # Merge list fields; use dict.fromkeys to deduplicate while preserving order
    all_treatments: list[str] = []
    all_imaging: list[str] = []
    for r in all_results:
        all_treatments.extend(r.get("conservative_treatment") or [])
        all_imaging.extend(r.get("prior_imaging") or [])

    combined_treatments = list(dict.fromkeys(all_treatments))
    combined_imaging = list(dict.fromkeys(all_imaging))

    # Take scalar fields from the first attachment that has them —
    # the first PDF is typically the physician's narrative letter.
    def _first_non_none(key: str) -> Any:
        for r in all_results:
            val = r.get(key)
            if val is not None:
                return val
        return None

    any_success = any(r.get("extraction_success") for r in all_results)
    warnings = [r["extraction_warning"] for r in all_results if r.get("extraction_warning")]

    combined: dict[str, Any] = {
        "extraction_success": any_success,
        "extraction_warning": "; ".join(warnings) if warnings else None,
        "raw_text": combined_raw_text,
        "patient_name": _first_non_none("patient_name"),
        "physician_name": _first_non_none("physician_name"),
        "diagnosis": _first_non_none("diagnosis"),
        "procedure_requested": _first_non_none("procedure_requested"),
        "conservative_treatment": combined_treatments,
        "treatment_duration": _first_non_none("treatment_duration"),
        "prior_imaging": combined_imaging,
        "medical_necessity": _first_non_none("medical_necessity"),
    }

    logger.info(
        "[node3] document_ingestion_node complete | "
        "raw_text_chars=%d, treatments=%d, imaging=%d",
        len(combined_raw_text),
        len(combined_treatments),
        len(combined_imaging),
    )
    return {"extracted_documents": combined}


# ---------------------------------------------------------------------------
# NODE 4: policy_retrieval_node
# ---------------------------------------------------------------------------


def policy_retrieval_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Retrieve relevant coverage policy chunks from Azure AI Search.

    Why this node exists:
    The clinical reasoning LLM must reason ONLY from retrieved policy documents
    rather than its training-data knowledge of coverage policies (which may be
    outdated or plan-specific).  Fetching the top-5 most relevant policy chunks
    here and injecting them into the LLM's system prompt in node 5 grounds the
    AI's decision in the actual, current policy text.

    Plan name extraction strategy:
    We look for the first ACTIVE insurance plan in member_context["insurance"].
    If no active plan is found we fall back to any plan, then to "private".
    This mirrors how real prior-auth requests work — the active plan at time
    of service governs coverage.

    Reads:  cpt_code, icd_code, member_context
    Sets:   retrieved_policies
    """
    cpt_code: str = state.get("cpt_code", "")
    icd_code: str = state.get("icd_code", "")
    member_context: dict = state.get("member_context") or {}

    logger.info(
        "[node4] policy_retrieval_node starting | CPT=%s ICD=%s",
        cpt_code, icd_code,
    )

    # --- Determine plan name from member insurance records ---
    insurance_records: list[dict] = member_context.get("insurance", [])
    plan_name = "private"  # safe default

    # Prefer the first active plan; fall back to first plan of any status
    active_plans = [p for p in insurance_records if p.get("is_active")]
    candidate = (active_plans or insurance_records or [{}])[0]
    if candidate.get("plan_name"):
        plan_name = candidate["plan_name"]
    elif candidate.get("ownership"):
        plan_name = candidate["ownership"]

    # Build a procedure description from extracted documents if available.
    # This improves procedure type detection in search_policies().
    extracted: dict = state.get("extracted_documents") or {}
    procedure_description: str = extracted.get("procedure_requested") or ""

    try:
        results = search_policies(
            cpt_code=cpt_code,
            icd_code=icd_code,
            plan_name=plan_name,
            procedure_description=procedure_description,
            top=5,
        )
        logger.info(
            "[node4] policy_retrieval_node complete | %d chunks retrieved | plan=%s",
            len(results),
            plan_name,
        )
    except Exception as exc:
        logger.error("[node4] Policy retrieval failed: %s", exc)
        results = []

    return {"retrieved_policies": results}


# ---------------------------------------------------------------------------
# NODE 5: clinical_reasoning_node
# ---------------------------------------------------------------------------


def _format_member_summary(member_context: dict) -> str:
    """Render member context as a compact text block for the LLM prompt.

    We deliberately exclude PII (exact address, full DOB) and keep the
    summary focused on clinically-relevant data the policy engine needs:
    age bracket, gender, active conditions, recent medications, imaging,
    and prior auth history.  Smaller prompts reduce latency and cost.
    """
    basic = member_context.get("basic_info") or {}
    lines: list[str] = []

    if basic:
        lines.append(f"Member: {basic.get('first_name', '')} {basic.get('last_name', '')} | "
                     f"Gender: {basic.get('gender', 'unknown')} | "
                     f"DOB: {basic.get('birthdate', 'unknown')}")

    # Active conditions — the most clinically relevant subset
    conditions = member_context.get("conditions") or []
    active_conds = [c for c in conditions if c.get("is_active")]
    if active_conds:
        cond_text = "; ".join(
            f"{c.get('code', '')} {c.get('description', '')}"
            for c in active_conds[:10]
        )
        lines.append(f"Active Conditions: {cond_text}")

    # Recent medications — step-therapy evidence
    medications = member_context.get("medications") or []
    if medications:
        med_text = "; ".join(
            m.get("description", "") for m in medications[:8] if m.get("description")
        )
        lines.append(f"Recent Medications: {med_text}")

    # Prior imaging — reduces duplicate imaging requests
    imaging = member_context.get("imaging") or []
    if imaging:
        img_text = "; ".join(
            f"{i.get('modality_description', '')} {i.get('bodysite_description', '')} ({i.get('study_date', '')})"
            for i in imaging[:5]
        )
        lines.append(f"Prior Imaging: {img_text}")

    # Prior auth history — precedent
    prior_auth = member_context.get("prior_auth_history") or []
    if prior_auth:
        pa_text = "; ".join(
            f"CPT {p.get('cpt_code', '')} → {p.get('status', '')} ({p.get('submitted_at', '')})"
            for p in prior_auth[:3]
        )
        lines.append(f"Prior Auth History: {pa_text}")

    return "\n".join(lines) if lines else "No member context available."


def _format_policy_chunks(policies: list[dict]) -> str:
    """Format retrieved policy chunks into a numbered list for the LLM prompt.

    Each chunk is separated by a divider and prefixed with its source filename
    so the LLM can cite a specific document in policy_basis.
    """
    if not policies:
        return "No policy documents retrieved."

    sections: list[str] = []
    for i, policy in enumerate(policies, start=1):
        header = f"[Policy {i} | Source: {policy.get('filename', 'unknown')} | Plan: {policy.get('plan_name', '')}]"
        sections.append(f"{header}\n{policy.get('content', '')}")

    return "\n\n".join(sections)


def clinical_reasoning_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Core LLM clinical reasoning node — produces the AI authorization decision.

    Why this design:
    - Temperature 0 for maximum determinism; coverage decisions must be
      reproducible and auditable.
    - JSON mode (response_format=json_object) guarantees parseable output
      without needing to parse markdown code blocks.
    - System prompt explicitly instructs the model to reason ONLY from the
      provided policy documents and clinical evidence — not from its training
      data knowledge of coverage policies, which could be stale or incorrect.
    - We validate the LLM output through ClinicalReasoningOutput (Pydantic)
      before trusting it, so schema violations surface as validation errors
      rather than downstream key errors.

    On validation failure: we produce a safe INSUFFICIENT_INFO result so the
    case is always routed to a nurse rather than silently dropped.

    Reads:  member_context, extracted_documents, retrieved_policies,
            clinical_notes, cpt_code, icd_code
    Sets:   reasoning_output, confidence_score
    """
    logger.info("[node5] clinical_reasoning_node starting")

    cpt_code: str = state.get("cpt_code", "")
    icd_code: str = state.get("icd_code", "")
    clinical_notes: str = state.get("clinical_notes", "")
    member_context: dict = state.get("member_context") or {}
    extracted_docs: dict = state.get("extracted_documents") or {}
    policies: list[dict] = state.get("retrieved_policies") or []

    # --- System prompt: role definition + strict output schema ---
    system_prompt = f"""You are a clinical policy engine for a health insurance prior authorization system.

Your ONLY job is to evaluate whether the submitted clinical evidence meets the
coverage criteria in the provided policy documents.

STRICT RULES:
1. Reason ONLY from the policy documents and clinical evidence provided below.
2. Do NOT use your training-data knowledge of medical coverage policies.
3. Do NOT invent criteria or make assumptions about what is "typically covered".
4. If the provided policy documents are insufficient to make a decision, use decision: INSUFFICIENT_INFO.
5. Every criterion_met and criterion_not_met must reference a specific policy section.

OUTPUT FORMAT — respond with a single valid JSON object matching this exact schema:
{{
  "decision": "APPROVE" | "DENY" | "INSUFFICIENT_INFO",
  "confidence_score": <float between 0.0 and 1.0>,
  "policy_basis": "<specific policy section citation, e.g. 'Coverage Policy CP-2024-0041 §3.2'>",
  "reasoning": "<plain English explanation, minimum 50 words, mapping evidence to criteria>",
  "missing_information": ["<item1>", ...],
  "criteria_met": ["<criterion1>", ...],
  "criteria_not_met": ["<criterion1>", ...]
}}

POLICY DOCUMENTS:
{_format_policy_chunks(policies)}"""

    # --- User message: the clinical evidence package ---
    # We combine clinical_notes (provider's written submission) with the
    # structured extraction from PDF attachments.  The attachment raw_text
    # is added only if it contains substantive content beyond the notes.
    evidence_sections: list[str] = [
        f"CPT CODE (procedure requested): {cpt_code}",
        f"ICD-10 CODE (diagnosis): {icd_code}",
        "",
        "=== MEMBER CLINICAL HISTORY ===",
        _format_member_summary(member_context),
        "",
        "=== PHYSICIAN CLINICAL NOTES ===",
        clinical_notes or "(none provided)",
    ]

    # Append extracted document content if it adds information beyond what
    # the physician already wrote in clinical_notes
    doc_raw = extracted_docs.get("raw_text", "")
    if doc_raw and len(doc_raw) > 100:
        evidence_sections += [
            "",
            "=== EXTRACTED ATTACHMENT CONTENT ===",
            doc_raw[:8000],  # cap at 8k chars to avoid token overflow
        ]

    # Structured fields from doc parser are appended as a quick-scan summary
    if extracted_docs.get("conservative_treatment"):
        evidence_sections += [
            "",
            "Conservative Treatments Documented in Attachments:",
            "\n".join(f"- {t}" for t in extracted_docs["conservative_treatment"]),
        ]
    if extracted_docs.get("prior_imaging"):
        evidence_sections += [
            "",
            "Prior Imaging Documented in Attachments:",
            "\n".join(f"- {img}" for img in extracted_docs["prior_imaging"]),
        ]
    if extracted_docs.get("medical_necessity"):
        evidence_sections += [
            "",
            "Medical Necessity Statement from Attachments:",
            extracted_docs["medical_necessity"],
        ]

    user_message = "\n".join(evidence_sections)

    # --- LLM call ---
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_json = response.choices[0].message.content or "{}"
        logger.info("[node5] LLM response received (%d chars)", len(raw_json))

    except Exception as exc:
        logger.error("[node5] LLM call failed: %s", exc)
        # Safe fallback: INSUFFICIENT_INFO routes case to human review
        fallback = {
            "decision": AIDecision.INSUFFICIENT_INFO.value,
            "confidence_score": 0.0,
            "policy_basis": "LLM call failed — unable to evaluate",
            "reasoning": f"Clinical reasoning could not be completed due to a system error: {exc}",
            "missing_information": ["AI system error — manual review required"],
            "criteria_met": [],
            "criteria_not_met": [],
        }
        return {"reasoning_output": fallback, "confidence_score": 0.0}

    # --- Parse and validate ---
    try:
        parsed = json.loads(raw_json)
        validated = ClinicalReasoningOutput(**parsed)
        reasoning_dict = validated.model_dump()
        confidence = validated.confidence_score
        logger.info(
            "[node5] clinical_reasoning_node complete | decision=%s confidence=%.2f",
            validated.decision,
            confidence,
        )
        return {"reasoning_output": reasoning_dict, "confidence_score": confidence}

    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error("[node5] LLM output validation failed: %s\nRaw: %s", exc, raw_json[:500])
        fallback = {
            "decision": AIDecision.INSUFFICIENT_INFO.value,
            "confidence_score": 0.0,
            "policy_basis": "Output validation failed — unable to parse AI response",
            "reasoning": f"The AI produced an invalid response format. Manual review required. Error: {exc}",
            "missing_information": ["AI output parse error — manual review required"],
            "criteria_met": [],
            "criteria_not_met": [],
        }
        return {"reasoning_output": fallback, "confidence_score": 0.0}


# ---------------------------------------------------------------------------
# NODE 6: confidence_evaluation_node
# ---------------------------------------------------------------------------


def confidence_evaluation_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Determine whether this case requires human nurse review.

    Why pure Python (no LLM):
    The routing decision must be fast, deterministic, and auditable.  Using
    another LLM call here would add latency and introduce non-determinism into
    a critical safety gate — we never want to accidentally auto-approve a case
    that should have been reviewed.

    When requires_human_review is True, the graph's conditional edge routes
    the workflow to human_review_node (node 7) before audit_output_node.
    When False, the workflow jumps directly to audit_output_node.

    Rules (any one triggers human review):
    1. confidence_score < CONFIDENCE_THRESHOLD (default 0.75)
    2. AI decision is INSUFFICIENT_INFO
    3. AI decision is DENY  — nurses must confirm all denials per policy
    4. is_urgent == True    — urgent cases always get a human touch
    5. retrieved_policies is empty — AI had no policy grounding to reason from

    Reads:  confidence_score, reasoning_output, is_urgent, retrieved_policies
    Sets:   requires_human_review
    """
    logger.info("[node6] confidence_evaluation_node starting")

    confidence: float = state.get("confidence_score") or 0.0
    reasoning: dict = state.get("reasoning_output") or {}
    is_urgent: bool = state.get("is_urgent", False)
    policies: list = state.get("retrieved_policies") or []

    decision: str = reasoning.get("decision", AIDecision.INSUFFICIENT_INFO.value)

    reasons: list[str] = []

    if confidence < _CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {confidence:.2f} < threshold {_CONFIDENCE_THRESHOLD}")

    if decision == AIDecision.INSUFFICIENT_INFO.value:
        reasons.append("AI decision is INSUFFICIENT_INFO")

    if decision == AIDecision.DENY.value:
        reasons.append("all denials require nurse confirmation")

    if is_urgent:
        reasons.append("case flagged as urgent")

    if not policies:
        reasons.append("no policy documents retrieved — AI had no grounding")

    requires_review = len(reasons) > 0

    if requires_review:
        logger.info(
            "[node6] Human review required. Reasons: %s",
            " | ".join(reasons),
        )
    else:
        logger.info(
            "[node6] No human review needed. "
            "confidence=%.2f decision=%s urgent=%s",
            confidence, decision, is_urgent,
        )

    logger.info(
        "[node6] confidence_evaluation_node complete | requires_human_review=%s",
        requires_review,
    )
    return {"requires_human_review": requires_review}


# ---------------------------------------------------------------------------
# NODE 7: human_review_node
# ---------------------------------------------------------------------------


def human_review_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Package the AI recommendation for nurse review and suspend the workflow.

    Why this design:
    LangGraph's interrupt mechanism pauses execution at this node and waits for
    an external event (the nurse submitting a decision via the API) before
    resuming.  This node's job is to package everything the nurse needs to make
    an informed decision into a single structured dict, then return it so it
    is persisted in the checkpointer state.

    After the nurse submits a decision via POST /cases/{case_id}/review, the
    graph resumes from this node's output with nurse_decision, nurse_id, and
    nurse_override_reason added to state, then proceeds to audit_output_node.

    The returned final_output is partial — audit_output_node will replace it
    with the complete summary after recording the nurse decision.

    Reads:  all state fields
    Sets:   final_output (partial handoff dict for nurse UI)
    """
    logger.info("[node7] human_review_node starting — packaging nurse handoff")

    reasoning: dict = state.get("reasoning_output") or {}
    member_context: dict = state.get("member_context") or {}
    basic_info: dict = member_context.get("basic_info") or {}

    # Build a concise summary of the member for the nurse review UI.
    # Nurses need to identify the patient and understand the clinical context
    # at a glance without opening the full EHR record.
    member_summary = {
        "member_id": state.get("member_id"),
        "name": f"{basic_info.get('first_name', '')} {basic_info.get('last_name', '')}".strip(),
        "date_of_birth": str(basic_info.get("birthdate", "unknown")),
        "gender": basic_info.get("gender", "unknown"),
        "active_conditions": [
            c.get("description", "") for c in
            (member_context.get("conditions") or [])
            if c.get("is_active")
        ][:5],
        "active_medications": [
            m.get("description", "") for m in
            (member_context.get("medications") or [])
        ][:5],
    }

    # The handoff dict surfaces everything the nurse needs to approve, deny,
    # or request more information without re-running the AI workflow.
    handoff = {
        "status": "PENDING_NURSE_REVIEW",
        "case_type": state.get("case_type"),
        "cpt_code": state.get("cpt_code"),
        "icd_code": state.get("icd_code"),
        "is_urgent": state.get("is_urgent"),
        "member_summary": member_summary,
        "ai_recommendation": reasoning.get("decision"),
        "confidence_score": state.get("confidence_score"),
        "policy_basis": reasoning.get("policy_basis"),
        "reasoning": reasoning.get("reasoning"),
        "criteria_met": reasoning.get("criteria_met", []),
        "criteria_not_met": reasoning.get("criteria_not_met", []),
        "missing_information": reasoning.get("missing_information", []),
        "nurse_decision": None,       # will be filled after nurse input
        "nurse_id": None,
        "nurse_override_reason": None,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "[node7] human_review_node complete — workflow suspended for nurse review | "
        "ai_recommendation=%s confidence=%.2f",
        reasoning.get("decision"),
        state.get("confidence_score") or 0.0,
    )
    return {"final_output": handoff}


# ---------------------------------------------------------------------------
# NODE 8: audit_output_node
# ---------------------------------------------------------------------------


def _map_to_case_status(
    ai_decision: str,
    nurse_decision: str | None,
    requires_human_review: bool,
) -> str:
    """Derive the final CaseStatus string from workflow outcomes.

    Priority: nurse decision > AI decision > pending state.
    """
    if nurse_decision:
        mapping = {
            "APPROVED": "APPROVED",
            "DENIED": "DENIED",
            "REQUEST_MORE_INFO": "PENDING_MORE_INFO",
        }
        return mapping.get(nurse_decision, "PENDING_NURSE_REVIEW")

    if requires_human_review:
        return "PENDING_NURSE_REVIEW"

    if ai_decision == AIDecision.APPROVE.value:
        return "APPROVED"
    if ai_decision == AIDecision.DENY.value:
        return "DENIED"
    return "PENDING_MORE_INFO"


def audit_output_node(state: PriorAuthState) -> PriorAuthStateUpdate:
    """Write the final case record to Supabase and assemble the API response.

    Why this node runs last:
    We write the authoritative record only after all decisions have been made
    (AI reasoning + optional nurse review).  Writing earlier would require
    UPDATE operations for every state change, increasing DB round-trips and
    complicating the audit trail.

    Case ID generation:
    We generate a UUID4 here (not at submission time) because the case_id is
    the primary key of the DB record.  Generating it here — after all processing
    — avoids orphaned records if the workflow fails before completion.

    On DB write failure:
    We still return final_output with audit_written=False so the API response
    is not blocked.  The ops team monitors audit_written=False records and can
    replay the write from the persisted LangGraph checkpoint state.

    Reads:  everything in state
    Sets:   case_id, audit_written, final_output
    """
    logger.info("[node8] audit_output_node starting")

    case_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)

    reasoning: dict = state.get("reasoning_output") or {}
    ai_decision: str = reasoning.get("decision", AIDecision.INSUFFICIENT_INFO.value)
    nurse_decision: str | None = state.get("nurse_decision")
    requires_human_review: bool = state.get("requires_human_review") or False
    is_urgent: bool = state.get("is_urgent", False)

    final_status = _map_to_case_status(ai_decision, nurse_decision, requires_human_review)

    # --- Write to prior_auth_requests table ---
    # Using psycopg2 directly (same pattern as member_service) — no ORM
    # overhead for a single INSERT.  Parameterised query prevents SQL injection.
    audit_written = False
    if _DATABASE_URL:
        try:
            conn = psycopg2.connect(_DATABASE_URL)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prior_auth_requests (
                        case_id, member_id, cpt_code, icd_code,
                        ai_recommendation, nurse_decision, status,
                        submitted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        case_id,
                        state.get("member_id"),
                        state.get("cpt_code"),
                        state.get("icd_code"),
                        ai_decision,
                        nurse_decision,
                        final_status,
                        now_utc,
                    ),
                )
            conn.commit()
            conn.close()
            audit_written = True
            logger.info("[node8] Case record written to DB | case_id=%s status=%s", case_id, final_status)
        except Exception as exc:
            logger.error(
                "[node8] DB write failed for case_id=%s: %s — continuing without persistence",
                case_id, exc,
            )
    else:
        logger.warning("[node8] DATABASE_URL not set; skipping DB write.")

    # --- Assemble final API response dict ---
    estimated_time = (
        "Within 72 hours (urgent)" if is_urgent else "Within 2 business days"
    )

    final_output: dict[str, Any] = {
        "case_id": case_id,
        "status": final_status,
        "member_id": state.get("member_id"),
        "ai_recommendation": ai_decision,
        "confidence_score": state.get("confidence_score"),
        "requires_human_review": requires_human_review,
        "nurse_decision": nurse_decision,
        "nurse_id": state.get("nurse_id"),
        "nurse_override_reason": state.get("nurse_override_reason"),
        "policy_basis": reasoning.get("policy_basis"),
        "criteria_met": reasoning.get("criteria_met", []),
        "criteria_not_met": reasoning.get("criteria_not_met", []),
        "missing_information": reasoning.get("missing_information", []),
        "submitted_at": now_utc.isoformat(),
        "estimated_decision_time": estimated_time,
        "audit_written": audit_written,
        # Validation error (member not found etc.) surfaced for API layer
        "validation_error": state.get("validation_error"),
        "message": (
            "Prior authorization request processed successfully."
            if not state.get("validation_error")
            else state.get("validation_error")
        ),
    }

    logger.info(
        "[node8] audit_output_node complete | case_id=%s status=%s audit_written=%s",
        case_id, final_status, audit_written,
    )
    return {
        "case_id": case_id,
        "audit_written": audit_written,
        "final_output": final_output,
    }
