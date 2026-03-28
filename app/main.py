"""
app/main.py

New workflow (as of restructure):
  Doctor submits  → case saved to DB immediately; no AI runs yet.
  Nurse triggers  → POST /prior-auth/{case_id}/analyze runs full LangGraph pipeline.
  Nurse decides   → POST /prior-auth/{case_id}/review records final decision.

Endpoints:
  POST /prior-auth/submit              — save case, return case_id instantly
  POST /prior-auth/{case_id}/analyze   — nurse triggers AI analysis (LangGraph)
  GET  /prior-auth/queue               — all cases for nurse queue view
  GET  /prior-auth/{case_id}/status    — status + full analysis output
  POST /prior-auth/{case_id}/review    — nurse decision; resumes interrupted graph
  GET  /health                         — liveness probe
  GET  /doctor-portal                  — doctor submission UI
  GET  /nurse-portal                   — nurse review portal UI
"""

from __future__ import annotations

import json
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


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
    provider_npi: str,
    clinical_notes: str,
    is_urgent: bool,
    case_type: str,
) -> bool:
    """INSERT a new case record with status PENDING_NURSE_REVIEW. Returns True on success."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prior_auth_requests (
                        case_id, thread_id, member_id, cpt_code, icd_code,
                        provider_npi, clinical_notes, is_urgent, case_type,
                        status, ai_analysis_status, submitted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        case_id, thread_id, member_id, cpt_code, icd_code,
                        provider_npi, clinical_notes, is_urgent, case_type,
                        "PENDING_NURSE_REVIEW", "NOT_RUN",
                        datetime.now(timezone.utc),
                    ),
                )
            conn.commit()
        return True
    except Exception as exc:
        logger.error("DB write failed for case_id=%s: %s", case_id, exc)
        return False


def _update_case_analysis(
    case_id: str,
    ai_recommendation: str | None,
    confidence_score: float | None,
    reasoning_output: dict | None,
    retrieved_policies: list | None,
) -> None:
    """Update case record with AI analysis results after /analyze runs."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE prior_auth_requests
                    SET ai_recommendation  = %s,
                        confidence_score   = %s,
                        reasoning_output   = %s,
                        retrieved_policies = %s,
                        ai_analysis_status = 'COMPLETE'
                    WHERE case_id = %s
                    """,
                    (
                        ai_recommendation,
                        confidence_score,
                        json.dumps(reasoning_output) if reasoning_output else None,
                        json.dumps(retrieved_policies) if retrieved_policies else None,
                        case_id,
                    ),
                )
            conn.commit()
    except Exception as exc:
        logger.error("Analysis update failed for case_id=%s: %s", case_id, exc)


def _update_thread_id(case_id: str, thread_id: str) -> None:
    """Set thread_id on a record."""
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
    """Patch case after nurse submits decision; mark ai_analysis_status COMPLETE."""
    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE prior_auth_requests
                    SET status             = %s,
                        nurse_decision     = %s,
                        ai_analysis_status = 'COMPLETE'
                    WHERE case_id = %s
                    """,
                    (status, nurse_decision, case_id),
                )
            conn.commit()
    except Exception as exc:
        logger.error("Post-review update failed for case_id=%s: %s", case_id, exc)


def _get_case_row(case_id: str) -> dict[str, Any] | None:
    """Fetch a single case row by case_id. Returns None if not found."""
    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT case_id, thread_id, member_id, cpt_code, icd_code,
                           provider_npi, clinical_notes, is_urgent, case_type,
                           ai_recommendation, nurse_decision, status, submitted_at,
                           ai_analysis_status, confidence_score,
                           reasoning_output, retrieved_policies
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


def _build_initial_state_from_row(row: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Build LangGraph initial state from a DB case row (used by /analyze endpoint)."""
    return {
        "raw_request": (
            f"Prior auth request: CPT {row.get('cpt_code')}, ICD {row.get('icd_code')}, "
            f"member {row.get('member_id')}, type {row.get('case_type', 'outpatient')}"
        ),
        "member_id": row.get("member_id", ""),
        "cpt_code": row.get("cpt_code", ""),
        "icd_code": row.get("icd_code", ""),
        "provider_npi": row.get("provider_npi", ""),
        "clinical_notes": row.get("clinical_notes", ""),
        "attachments": [],
        "is_urgent": bool(row.get("is_urgent", False)),
        "case_type": row.get("case_type", "outpatient"),
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
        ai_recommendation=(
            row.get("ai_recommendation")
            or state_values.get("reasoning_output", {}).get("decision")
        ),
        confidence_score=row.get("confidence_score") or state_values.get("confidence_score"),
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


@app.get("/doctor-portal")
async def doctor_portal():
    return FileResponse("frontend/doctor-portal.html")


@app.get("/nurse-portal")
async def nurse_portal():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "prior-auth-ai"}


@app.post("/prior-auth/submit", response_model=PriorAuthResponse)
async def submit_prior_auth(request: PriorAuthRequest):
    """Save a new prior authorization request to the database immediately.

    Does NOT run AI analysis. Returns a case_id instantly so the doctor
    gets confirmation of receipt. The nurse triggers analysis separately
    via POST /prior-auth/{case_id}/analyze.
    """
    case_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    logger.info(
        "POST /prior-auth/submit | member_id=%s case_id=%s",
        request.member_id, case_id,
    )

    success = _write_case_record(
        case_id=case_id,
        thread_id=thread_id,
        member_id=request.member_id,
        cpt_code=request.cpt_code,
        icd_code=request.icd_code,
        provider_npi=request.provider_npi,
        clinical_notes=request.clinical_notes,
        is_urgent=request.is_urgent,
        case_type=request.case_type.value,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save prior authorization request.")

    estimated_time = "Within 72 hours (urgent)" if request.is_urgent else "Within 2 business days"

    return PriorAuthResponse(
        case_id=case_id,
        status=CaseStatus.PENDING_NURSE_REVIEW,
        message="Prior authorization request received. Awaiting nurse-triggered AI analysis.",
        estimated_decision_time=estimated_time,
    )


@app.post("/prior-auth/{case_id}/analyze")
async def analyze_prior_auth(case_id: str):
    """Run the full LangGraph AI analysis for a case. Called by the nurse.

    Fetches the case from DB, runs the AI workflow (member context fetch,
    policy retrieval, clinical reasoning, confidence evaluation), stores
    all results in DB, and returns the full analysis so the nurse can
    read the policy language and AI reasoning before deciding.
    """
    logger.info("POST /prior-auth/%s/analyze", case_id)

    row = _get_case_row(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    if row.get("ai_analysis_status") == "COMPLETE":
        raise HTTPException(
            status_code=409,
            detail="AI analysis has already been run for this case.",
        )

    thread_id = row.get("thread_id") or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        initial_state = _build_initial_state_from_row(row, thread_id)
        state_values = _stream_to_completion(initial_state, config)
    except Exception as exc:
        logger.error("Graph execution failed for case_id=%s: %s", case_id, exc)
        raise HTTPException(status_code=500, detail="AI analysis failed. Please try again.")

    reasoning_output = state_values.get("reasoning_output") or {}
    retrieved_policies = state_values.get("retrieved_policies") or []
    confidence_score = state_values.get("confidence_score")
    ai_recommendation = reasoning_output.get("decision")

    _update_case_analysis(
        case_id=case_id,
        ai_recommendation=ai_recommendation,
        confidence_score=float(confidence_score) if confidence_score is not None else None,
        reasoning_output=reasoning_output,
        retrieved_policies=retrieved_policies if isinstance(retrieved_policies, list) else [],
    )

    logger.info(
        "Analysis complete | case_id=%s decision=%s confidence=%s",
        case_id, ai_recommendation, confidence_score,
    )

    return {
        "case_id": case_id,
        "decision": ai_recommendation,
        "confidence_score": float(confidence_score) if confidence_score is not None else None,
        "policy_basis": reasoning_output.get("policy_basis"),
        "reasoning": reasoning_output.get("reasoning"),
        "criteria_met": reasoning_output.get("criteria_met", []),
        "criteria_not_met": reasoning_output.get("criteria_not_met", []),
        "missing_information": reasoning_output.get("missing_information", []),
        "retrieved_policies": retrieved_policies if isinstance(retrieved_policies, list) else [],
    }


@app.get("/members/samples")
async def get_sample_members():
    """Return 10 random members for demo sample data in the doctor portal."""
    try:
        with _get_db() as conn:
            with conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute("""
                    SELECT
                        m.id as member_id,
                        m.first_name,
                        m.last_name,
                        m.birthdate,
                        m.gender,
                        m.city,
                        m.state
                    FROM members m
                    ORDER BY RANDOM()
                    LIMIT 10
                """)
                rows = cur.fetchall()
                members = []
                for row in rows:
                    d = dict(row)
                    if d.get("birthdate") and hasattr(d["birthdate"], "isoformat"):
                        d["birthdate"] = d["birthdate"].isoformat()
                    members.append(d)
                return {"members": members}
    except Exception as exc:
        logger.error("Sample members fetch failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch sample members.",
        )


@app.get("/prior-auth/queue")
async def get_case_queue():
    """Return all cases sorted for the nurse queue.

    Order: PENDING_NURSE_REVIEW first (newest first), then APPROVED,
    DENIED, PENDING_MORE_INFO. Includes member first/last name via
    LEFT JOIN so the nurse can see patient name without a separate lookup.
    """
    try:
        with _get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        p.case_id,
                        p.member_id,
                        p.cpt_code,
                        p.icd_code,
                        p.status,
                        p.ai_analysis_status,
                        p.is_urgent,
                        p.submitted_at,
                        p.nurse_decision,
                        p.ai_recommendation,
                        m.first_name,
                        m.last_name
                    FROM prior_auth_requests p
                    LEFT JOIN members m ON p.member_id::text = m.id::text
                    ORDER BY
                        CASE p.status
                            WHEN 'PENDING_NURSE_REVIEW' THEN 0
                            WHEN 'APPROVED'             THEN 1
                            WHEN 'DENIED'               THEN 2
                            WHEN 'PENDING_MORE_INFO'    THEN 3
                            ELSE 4
                        END,
                        p.submitted_at DESC
                    """
                )
                rows = cur.fetchall()
                cases = []
                for row in rows:
                    d = dict(row)
                    if d.get("submitted_at") and hasattr(d["submitted_at"], "isoformat"):
                        d["submitted_at"] = d["submitted_at"].isoformat()
                    cases.append(d)
                return {"cases": cases, "total": len(cases)}
    except Exception as exc:
        logger.error("Queue fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch case queue.")


@app.get("/prior-auth/{case_id}/status")
async def get_case_status(case_id: str):
    """Return the current status of a prior authorization case.

    Includes reasoning_output and retrieved_policies stored in DB after
    /analyze runs, so the nurse can read full policy language and AI
    reasoning from a single endpoint.
    """
    logger.info("GET /prior-auth/%s/status", case_id)

    row = _get_case_row(case_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    # Parse JSONB columns — psycopg2 may return them as dict (native) or str
    reasoning_output = row.get("reasoning_output") or {}
    if isinstance(reasoning_output, str):
        try:
            reasoning_output = json.loads(reasoning_output)
        except Exception:
            reasoning_output = {}

    retrieved_policies = row.get("retrieved_policies") or []
    if isinstance(retrieved_policies, str):
        try:
            retrieved_policies = json.loads(retrieved_policies)
        except Exception:
            retrieved_policies = []

    status_str = row.get("status", "PENDING_NURSE_REVIEW")
    try:
        case_status = CaseStatus(status_str)
    except ValueError:
        case_status = CaseStatus.PENDING_NURSE_REVIEW

    submitted_at = row.get("submitted_at")
    if submitted_at and hasattr(submitted_at, "isoformat"):
        submitted_at = submitted_at.isoformat()

    return {
        "case_id": row["case_id"],
        "status": status_str,
        "member_id": row.get("member_id", ""),
        "cpt_code": row.get("cpt_code"),
        "icd_code": row.get("icd_code"),
        "provider_npi": row.get("provider_npi"),
        "clinical_notes": row.get("clinical_notes"),
        "is_urgent": row.get("is_urgent"),
        "case_type": row.get("case_type"),
        "ai_recommendation": row.get("ai_recommendation"),
        "confidence_score": row.get("confidence_score"),
        "ai_analysis_status": row.get("ai_analysis_status", "NOT_RUN"),
        "requires_human_review": case_status == CaseStatus.PENDING_NURSE_REVIEW,
        "submitted_at": submitted_at,
        "nurse_decision": row.get("nurse_decision"),
        "reasoning_output": reasoning_output,
        "retrieved_policies": retrieved_policies,
    }


@app.post("/prior-auth/{case_id}/review")
async def submit_nurse_review(case_id: str, nurse_input: NurseDecision):
    logger.info(
        "POST /prior-auth/%s/review | nurse_id=%s decision=%s",
        case_id, nurse_input.nurse_id, nurse_input.decision,
    )

    row = _get_case_row(case_id)
    if not row:
        raise HTTPException(status_code=404,
            detail=f"Case '{case_id}' not found.")

    current_status = row.get("status", "")
    if current_status not in (
        "PENDING_NURSE_REVIEW",
        "PENDING_AI_REVIEW",
    ):
        raise HTTPException(status_code=409,
            detail=f"Case status is '{current_status}' and cannot be reviewed.")

    status_map = {
        "APPROVED": "APPROVED",
        "DENIED": "DENIED",
        "REQUEST_MORE_INFO": "PENDING_MORE_INFO",
    }
    final_status = status_map.get(
        nurse_input.decision.value, "PENDING_NURSE_REVIEW"
    )

    try:
        with _get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE prior_auth_requests
                    SET status = %s,
                        nurse_decision = %s,
                        nurse_id = %s,
                        ai_analysis_status = 'COMPLETE'
                    WHERE case_id = %s
                """, (
                    final_status,
                    nurse_input.decision.value,
                    nurse_input.nurse_id,
                    case_id,
                ))
            conn.commit()
    except Exception as exc:
        logger.error("Review update failed for case_id=%s: %s", case_id, exc)
        raise HTTPException(status_code=500,
            detail="Failed to save decision.")

    logger.info(
        "Nurse review complete | case_id=%s status=%s nurse=%s",
        case_id, final_status, nurse_input.nurse_id,
    )

    updated_row = _get_case_row(case_id) or row
    submitted_at = updated_row.get("submitted_at")
    if submitted_at and hasattr(submitted_at, "isoformat"):
        submitted_at = submitted_at.isoformat()

    return {
        "case_id": str(case_id),
        "status": str(final_status),
        "member_id": str(updated_row.get("member_id", "")),
        "nurse_decision": str(nurse_input.decision.value),
        "nurse_id": str(nurse_input.nurse_id),
        "ai_recommendation": str(updated_row.get("ai_recommendation", "") or ""),
        "submitted_at": str(submitted_at or ""),
        "message": f"Decision recorded: {nurse_input.decision.value}",
    }
