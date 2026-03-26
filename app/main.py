"""
app/main.py

FastAPI HTTP entry point for the Prior Authorization AI system.

Exposes the LangGraph workflow as a REST API with four functional endpoints:
  POST /prior-auth/submit          — run workflow, return case reference
  GET  /prior-auth/{case_id}/status — poll case status
  POST /prior-auth/{case_id}/review — nurse decision; resumes interrupted graph
  GET  /health                      — liveness probe

DB record lifecycle (coordinates with audit_output_node):
  - For cases that complete without human review: audit_output_node writes the
    DB record; main.py updates it to add thread_id so the review endpoint can
    look it up later.
  - For cases that are interrupted before human_review: audit_output_node has
    NOT run yet, so main.py writes a preliminary record using thread_id as the
    case_id.  After the nurse submits a decision and the graph resumes, main.py
    patches the preliminary record with the final decision and status.

This design keeps the graph nodes free of HTTP concerns while ensuring every
case always has a queryable DB record from the moment /submit returns.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.graph.graph import graph_app
from app.models.schemas import (
    CaseStatus,
    CaseStatusResponse,
    NurseDecision,
    PriorAuthRequest,
    PriorAuthResponse,
)

load_dotenv()

logger = logging.getLogger(__name__)

_DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prior Authorization AI System",
    description=(
        "AI-assisted prior authorization workflow for healthcare payers. "
        "Combines LangGraph orchestration, Azure AI Search RAG, and GPT-4o "
        "clinical reasoning to accelerate coverage decisions."
    ),
    version="1.0.0",
)

# Allow all origins for local frontend / EHR development.
# Tighten to specific origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database helpers — same psycopg2 pattern as member_service.py
# ---------------------------------------------------------------------------


@contextmanager
def _get_db():
    """Yield a psycopg2 connection and guarantee cleanup."""
    conn = None
    try:
        conn = psycopg2.connect(_DATABASE_URL)
        yield conn
    finally:
        if conn and not conn.closed:
            conn.close()


def _write_case_record(
    case_id: str,
    thread_id: str,
    member_id: str,
    cpt_code: str,
    icd_code: str,
    ai_recommendation: str | None,
    nurse_decision: str | None,
    status: str,
) -> bool:
    """INSERT a new case record.  Returns True on success."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prior_auth_requests (
                        case_id, thread_id, member_id, cpt_code, icd_code,
                        ai_recommendation, nurse_decision, status, submitted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        case_id, thread_id, member_id, cpt_code, icd_code,
                        ai_recommendation, nurse_decision, status,
                        datetime.now(timezone.utc),
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        logger.error("DB write failed for case_id=%s: %s", case_id, exc)
        return False


def _update_thread_id(case_id: str, thread_id: str) -> None:
    """Set thread_id on a record written by audit_output_node."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE prior_auth_requests SET thread_id = %s WHERE case_id = %s",
                    (thread_id, case_id),
                )
            conn.commit()
    except Exception as exc:
        logger.error("thread_id update failed for case_id=%s: %s", case_id, exc)


def _update_case_after_review(
    case_id: str,
    status: str,
    nurse_decision: str | None,
    nurse_id: str | None,
) -> None:
    """Patch the preliminary record after the graph resumes and finishes."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE prior_auth_requests
                    SET status = %s, nurse_decision = %s
                    WHERE case_id = %s
                    """,
                    (status, nurse_decision, case_id),
                )
            conn.commit()
    except Exception as exc:
        logger.error("Post-review update failed for case_id=%s: %s", case_id, exc)


def _get_case_row(case_id: str) -> dict[str, Any] | None:
    """Fetch a single case row by case_id.  Returns None if not found."""
    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT case_id, thread_id, member_id, cpt_code, icd_code,
                           ai_recommendation, nurse_decision, status, submitted_at
                    FROM prior_auth_requests
                    WHERE case_id = %s
                    LIMIT 1
                    """,
                    (case_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as exc:
        logger.error("DB lookup failed for case_id=%s: %s", case_id, exc)
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_initial_state(request: PriorAuthRequest, thread_id: str) -> dict[str, Any]:
    """Map a validated PriorAuthRequest into the LangGraph initial state dict."""
    return {
        "raw_request": (
            f"Prior auth request: CPT {request.cpt_code}, ICD {request.icd_code}, "
            f"member {request.member_id}, type {request.case_type.value}"
        ),
        "member_id": request.member_id,
        "cpt_code": request.cpt_code,
        "icd_code": request.icd_code,
        "provider_npi": request.provider_npi,
        "clinical_notes": request.clinical_notes,
        "attachments": request.attachments,
        "is_urgent": request.is_urgent,
        "case_type": request.case_type.value,
        # Optional fields initialised to None so the TypedDict is complete
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


def _stream_to_completion(initial_or_update: dict, config: dict) -> dict[str, Any]:
    """Run graph_app.stream() until interrupt or END and return the snapshot values."""
    for _ in graph_app.stream(initial_or_update, config=config):
        pass  # drain the stream; each step updates the checkpointer in-place
    return graph_app.get_state(config).values


def _build_case_status_response(
    row: dict[str, Any],
    state_values: dict[str, Any],
) -> CaseStatusResponse:
    """Construct a CaseStatusResponse from DB row + LangGraph state values."""
    nurse_decision_obj: NurseDecision | None = None
    nurse_str = row.get("nurse_decision") or state_values.get("nurse_decision")
    nurse_id = state_values.get("nurse_id")
    if nurse_str and nurse_id:
        try:
            nurse_decision_obj = NurseDecision(
                decision=nurse_str,
                nurse_id=nurse_id,
                override_reason=state_values.get("nurse_override_reason"),
            )
        except Exception:
            pass  # malformed nurse data — surface None rather than 500

    status_str = row.get("status", "PENDING_AI_REVIEW")
    try:
        case_status = CaseStatus(status_str)
    except ValueError:
        case_status = CaseStatus.PENDING_AI_REVIEW

    requires_review: bool = bool(
        state_values.get("requires_human_review")
        or case_status == CaseStatus.PENDING_NURSE_REVIEW
    )

    submitted_at = row.get("submitted_at") or datetime.now(timezone.utc)

    return CaseStatusResponse(
        case_id=row["case_id"],
        status=case_status,
        member_id=row.get("member_id", ""),
        ai_recommendation=row.get("ai_recommendation") or state_values.get("reasoning_output", {}).get("decision"),
        confidence_score=state_values.get("confidence_score"),
        requires_human_review=requires_review,
        submitted_at=submitted_at,
        nurse_decision=nurse_decision_obj,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "service": "Prior Authorization AI System",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "prior-auth-ai"}


@app.post("/prior-auth/submit", response_model=PriorAuthResponse)
async def submit_prior_auth(request: PriorAuthRequest):
    """Submit a new prior authorization request.

    Launches the LangGraph workflow and streams it until it either completes
    or pauses for nurse review.  Returns a case reference immediately so the
    provider can poll for status.

    Two completion paths:
    1. Auto-decided (high confidence APPROVE / no human review needed):
       audit_output_node runs during this call, writes the DB record, and we
       update it with thread_id before returning.
    2. Interrupted for human review:
       audit_output_node has NOT run yet.  We write a preliminary DB record
       using thread_id as the case_id so the caller can poll status and the
       /review endpoint can look up the right LangGraph thread.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("POST /prior-auth/submit | member_id=%s thread_id=%s", request.member_id, thread_id)

    try:
        initial_state = _build_initial_state(request, thread_id)
        state_values = _stream_to_completion(initial_state, config)
    except Exception as exc:
        logger.error("Graph execution failed for thread_id=%s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail="Prior authorization processing failed. Please try again.")

    snapshot = graph_app.get_state(config)
    interrupted = bool(snapshot.next and "human_review" in snapshot.next)

    is_urgent = request.is_urgent
    estimated_time = "Within 72 hours (urgent)" if is_urgent else "Within 2 business days"

    if interrupted:
        # The graph is paused before human_review. audit_output hasn't run.
        # Use thread_id as the preliminary case_id so status polling works.
        case_id = thread_id
        status = CaseStatus.PENDING_NURSE_REVIEW
        message = (
            "Prior authorization request received. "
            "Clinical review by a licensed nurse is required before a decision can be issued."
        )
        logger.info("Graph interrupted for nurse review | thread_id=%s", thread_id)

        _write_case_record(
            case_id=case_id,
            thread_id=thread_id,
            member_id=request.member_id,
            cpt_code=request.cpt_code,
            icd_code=request.icd_code,
            ai_recommendation=(state_values.get("reasoning_output") or {}).get("decision"),
            nurse_decision=None,
            status=status.value,
        )

    else:
        # Graph completed fully. audit_output_node wrote the DB record.
        # Retrieve the case_id it generated and link our thread_id to it.
        case_id = state_values.get("case_id")
        final_output = state_values.get("final_output") or {}

        if not case_id:
            # audit_output_node failed silently — fall back to thread_id
            logger.warning("No case_id in completed state; falling back to thread_id=%s", thread_id)
            case_id = thread_id
            _write_case_record(
                case_id=case_id,
                thread_id=thread_id,
                member_id=request.member_id,
                cpt_code=request.cpt_code,
                icd_code=request.icd_code,
                ai_recommendation=final_output.get("ai_recommendation"),
                nurse_decision=None,
                status=final_output.get("status", CaseStatus.PENDING_AI_REVIEW.value),
            )
        else:
            _update_thread_id(case_id, thread_id)

        raw_status = final_output.get("status", CaseStatus.PENDING_AI_REVIEW.value)
        try:
            status = CaseStatus(raw_status)
        except ValueError:
            status = CaseStatus.PENDING_AI_REVIEW

        validation_error = state_values.get("validation_error")
        if validation_error:
            message = validation_error
        else:
            message = final_output.get("message", "Prior authorization request processed successfully.")

        logger.info("Graph completed | case_id=%s status=%s", case_id, status)

    return PriorAuthResponse(
        case_id=case_id,
        status=status,
        message=message,
        estimated_decision_time=estimated_time,
    )


@app.get("/prior-auth/{case_id}/status", response_model=CaseStatusResponse)
async def get_case_status(case_id: str):
    """Return the current status of a prior authorization case.

    Combines the DB record (authoritative status and decision strings) with
    the LangGraph checkpoint state (confidence score, full reasoning context)
    to produce a complete CaseStatusResponse.
    """
    logger.info("GET /prior-auth/%s/status", case_id)

    row = _get_case_row(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    # Enrich with checkpoint state if the thread is still accessible
    state_values: dict[str, Any] = {}
    thread_id = row.get("thread_id")
    if thread_id:
        try:
            snapshot = graph_app.get_state({"configurable": {"thread_id": thread_id}})
            state_values = snapshot.values or {}
        except Exception as exc:
            logger.warning("Could not retrieve checkpoint for thread_id=%s: %s", thread_id, exc)

    try:
        return _build_case_status_response(row, state_values)
    except Exception as exc:
        logger.error("Failed to build CaseStatusResponse for case_id=%s: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve case status.")


@app.post("/prior-auth/{case_id}/review", response_model=CaseStatusResponse)
async def submit_nurse_review(case_id: str, nurse_input: NurseDecision):
    """Record a nurse's review decision and resume the LangGraph workflow.

    Looks up the LangGraph thread_id stored in the DB for this case, resumes
    the interrupted workflow with the nurse's decision, then patches the case
    record with the final status and decision string.

    The nurse must provide:
    - decision: APPROVED | DENIED | REQUEST_MORE_INFO
    - nurse_id: licensed clinician credential ID (required for HIPAA audit)
    - override_reason: required if decision differs from AI recommendation
    """
    logger.info(
        "POST /prior-auth/%s/review | nurse_id=%s decision=%s",
        case_id, nurse_input.nurse_id, nurse_input.decision,
    )

    row = _get_case_row(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    thread_id = row.get("thread_id")
    if not thread_id:
        raise HTTPException(
            status_code=409,
            detail="This case has no active workflow thread and cannot be reviewed.",
        )

    # Verify the case is actually waiting for review before resuming
    current_status = row.get("status", "")
    if current_status not in (
        CaseStatus.PENDING_NURSE_REVIEW.value,
        CaseStatus.PENDING_AI_REVIEW.value,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Case is in status '{current_status}' and is not awaiting nurse review.",
        )

    config = {"configurable": {"thread_id": thread_id}}

    # Resume the LangGraph workflow.
    # The graph was interrupted before human_review_node; it continues from
    # that point with the nurse decision injected into state.
    nurse_update: dict[str, Any] = {
        "nurse_decision": nurse_input.decision.value,
        "nurse_id": nurse_input.nurse_id,
        "nurse_override_reason": nurse_input.override_reason,
    }

    try:
        state_values = _stream_to_completion(nurse_update, config)
    except Exception as exc:
        logger.error("Graph resume failed for case_id=%s thread_id=%s: %s", case_id, thread_id, exc)
        raise HTTPException(status_code=500, detail="Failed to process nurse review. Please try again.")

    # audit_output_node ran during the resume and wrote its own record with a
    # different case_id.  We patch the ORIGINAL record (the one the caller
    # knows about) with the final status and nurse decision so /status works.
    final_output = state_values.get("final_output") or {}
    final_status = final_output.get("status", CaseStatus.PENDING_NURSE_REVIEW.value)

    _update_case_after_review(
        case_id=case_id,
        status=final_status,
        nurse_decision=nurse_input.decision.value,
        nurse_id=nurse_input.nurse_id,
    )

    logger.info(
        "Nurse review complete | case_id=%s final_status=%s nurse_id=%s",
        case_id, final_status, nurse_input.nurse_id,
    )

    # Return the updated status of the original case record
    updated_row = _get_case_row(case_id) or row
    updated_row["nurse_decision"] = nurse_input.decision.value

    try:
        return _build_case_status_response(updated_row, state_values)
    except Exception as exc:
        logger.error("Failed to build post-review CaseStatusResponse for %s: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="Review recorded but failed to build response.")
