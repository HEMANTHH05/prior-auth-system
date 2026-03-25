"""
Pydantic v2 schemas for the Prior Authorization AI system.

Layered design:
  - PriorAuthRequest      → FastAPI ingestion (HTTP boundary)
  - ClinicalReasoningOutput → LangGraph node output (LLM boundary)
  - NurseDecision          → Nurse review input (human-in-the-loop boundary)
  - CaseStatusResponse     → Case status polling (API response)
  - PriorAuthResponse      → Submission acknowledgement (API response)
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------


class AIDecision(str, Enum):
    """Three-way decision produced by the clinical-reasoning LLM node.

    APPROVE           — criteria clearly met; recommend approval.
    DENY              — criteria clearly not met; recommend denial.
    INSUFFICIENT_INFO — not enough clinical evidence to decide; escalate to
                        nurse or request more information from the provider.
    """

    APPROVE = "APPROVE"
    DENY = "DENY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"


class NurseDecisionEnum(str, Enum):
    """Final disposition recorded after a human nurse reviews the AI output.

    APPROVED           — nurse confirms or overrides AI to approve.
    DENIED             — nurse confirms or overrides AI to deny.
    REQUEST_MORE_INFO  — nurse puts the case on hold pending additional
                         clinical documentation from the provider.
    """

    APPROVED = "APPROVED"
    DENIED = "DENIED"
    REQUEST_MORE_INFO = "REQUEST_MORE_INFO"


class CaseStatus(str, Enum):
    """Lifecycle states a prior-auth case can be in.

    PENDING_AI_REVIEW      → initial state after submission.
    PENDING_NURSE_REVIEW   → AI has made a recommendation; awaiting human.
    APPROVED               → final approved decision.
    DENIED                 → final denied decision.
    PENDING_MORE_INFO      → case on hold; provider must supply more docs.
    """

    PENDING_AI_REVIEW = "PENDING_AI_REVIEW"
    PENDING_NURSE_REVIEW = "PENDING_NURSE_REVIEW"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING_MORE_INFO = "PENDING_MORE_INFO"


class CaseType(str, Enum):
    """Clinical context of the authorization request.

    Controls which policy document set is retrieved during RAG and which
    coverage rules apply.
    """

    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    MEDICATION = "medication"
    DME = "dme"          # Durable Medical Equipment
    BEHAVIORAL = "behavioral"


# ---------------------------------------------------------------------------
# 1. PriorAuthRequest — HTTP ingestion layer
# ---------------------------------------------------------------------------


class PriorAuthRequest(BaseModel):
    """Validated input received from the provider/EHR via the FastAPI endpoint.

    This is the entry point into the system. All fields are validated at the
    HTTP boundary so downstream LangGraph nodes can trust the data.
    """

    member_id: Annotated[str, Field(
        min_length=1,
        max_length=50,
        description="Health-plan member identifier. Must match the insurer's "
                    "member roster. Example: 'MBR-00123456'.",
        examples=["MBR-00123456"],
    )]
    """Identifies the patient whose coverage will be checked against policy."""

    cpt_code: Annotated[str, Field(
        description="Current Procedural Terminology code for the requested "
                    "service. Must be exactly 5 digits (e.g., '27447' for "
                    "total knee arthroplasty).",
        examples=["27447"],
        pattern=r"^\d{5}$",
    )]
    """Used to look up the specific coverage policy and medical necessity
    criteria that apply to this procedure."""

    icd_code: Annotated[str, Field(
        description="ICD-10-CM diagnosis code justifying the procedure. "
                    "Format: letter + 2 digits, optional dot, optional "
                    "alphanumeric suffix (e.g., 'M17.11'). "
                    "ICD-9 is NOT accepted.",
        examples=["M17.11", "Z23", "J18.9"],
        pattern=r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$",
    )]
    """Drives medical-necessity determination: the diagnosis must align with
    the CPT code's coverage criteria in the policy documents."""

    provider_npi: Annotated[str, Field(
        description="10-digit National Provider Identifier of the requesting "
                    "physician or facility. Used to verify network status and "
                    "provider eligibility.",
        examples=["1234567890"],
        pattern=r"^\d{10}$",
    )]
    """Required by CMS; also used to check whether the provider is in-network
    for the member's plan, which affects coverage rules."""

    clinical_notes: Annotated[str, Field(
        min_length=20,
        max_length=50_000,
        description="Free-text clinical documentation: H&P, operative notes, "
                    "progress notes, or treatment history. This is the primary "
                    "input to the LLM's clinical reasoning step.",
        examples=["Patient presents with severe bilateral knee osteoarthritis "
                  "(Kellgren-Lawrence grade 4). Conservative therapy including "
                  "PT and NSAIDs has failed over 6 months. Requesting TKA."],
    )]
    """Embedded into the LLM prompt verbatim; quality here directly determines
    AI decision confidence."""

    attachments: list[str] = Field(
        default_factory=list,
        description="Azure Blob Storage URLs for supporting documents "
                    "(imaging reports, lab results, prior auth letters). "
                    "Processed by the document-extraction node before the "
                    "clinical-reasoning node runs.",
        examples=[["https://blob.core.windows.net/docs/mri-report.pdf"]],
    )
    """Supplemental evidence; extracted text is appended to the LLM context
    window alongside clinical_notes."""

    is_urgent: bool = Field(
        default=False,
        description="True if this is an urgent/expedited review request. "
                    "Urgent cases bypass standard queue prioritisation and "
                    "must be decided within 72 hours per CMS rules.",
    )
    """Affects SLA enforcement and queue ordering in the workflow engine."""

    case_type: CaseType = Field(
        default=CaseType.OUTPATIENT,
        description="Clinical category of the request. Controls policy "
                    "retrieval scope in the RAG pipeline.",
        examples=["outpatient"],
    )
    """Routed to the appropriate policy index in Azure AI Search."""

    @field_validator("icd_code", mode="before")
    @classmethod
    def normalize_icd_code(cls, v: str) -> str:
        """Uppercase and strip whitespace so 'm17.11' and 'M17.11' both pass."""
        return v.strip().upper()

    @field_validator("cpt_code", "provider_npi", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "member_id": "MBR-00123456",
                "cpt_code": "27447",
                "icd_code": "M17.11",
                "provider_npi": "1234567890",
                "clinical_notes": (
                    "Patient is a 68-year-old with severe bilateral knee OA "
                    "(KL grade 4). Six months of conservative treatment "
                    "(PT, NSAIDs, cortisone injections) has failed. "
                    "Requesting total knee arthroplasty."
                ),
                "attachments": [],
                "is_urgent": False,
                "case_type": "outpatient",
            }
        }
    }


# ---------------------------------------------------------------------------
# 2. ClinicalReasoningOutput — LangGraph node output (LLM boundary)
# ---------------------------------------------------------------------------


class ClinicalReasoningOutput(BaseModel):
    """Structured output produced by the clinical-reasoning LangGraph node.

    The LLM is instructed to respond strictly in this schema (via structured
    output / function calling). Pydantic validates the parsed response before
    it is written to the database and surfaced to the nurse review node.
    """

    decision: AIDecision = Field(
        description="The LLM's recommended authorization decision. "
                    "One of APPROVE | DENY | INSUFFICIENT_INFO.",
    )
    """Primary output consumed by the nurse review node and stored as
    ai_recommendation on the case record."""

    confidence_score: Annotated[float, Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in its own decision, expressed as a "
                    "probability in [0.0, 1.0]. Scores below 0.75 "
                    "automatically trigger human review regardless of decision.",
        examples=[0.87],
    )]
    """Drives the requires_human_review flag on the case. Low confidence is a
    signal that the clinical notes are ambiguous or incomplete."""

    policy_basis: str = Field(
        min_length=10,
        description="Citation of the specific policy section(s) the decision "
                    "is grounded in, e.g. 'Coverage Policy CP-2024-0041 §3.2 "
                    "– Medical Necessity Criteria for TKA'.",
        examples=["Coverage Policy CP-2024-0041 §3.2"],
    )
    """Regulatory audit trail: nurses and auditors must be able to trace every
    AI decision back to a policy document."""

    reasoning: str = Field(
        min_length=50,
        description="Plain-English explanation of how the clinical evidence "
                    "maps to (or fails to meet) the policy criteria. Must be "
                    "specific enough for a nurse to validate or override.",
    )
    """Surfaced verbatim in the nurse review UI so reviewers understand the
    AI's logic without re-reading source documents."""

    missing_information: list[str] = Field(
        default_factory=list,
        description="List of specific clinical data points absent from the "
                    "submitted notes that would be needed for a definitive "
                    "decision. Populated when decision is INSUFFICIENT_INFO.",
        examples=[["Conservative therapy duration", "BMI documentation"]],
    )
    """Drives the 'request more info' message sent back to the provider."""

    criteria_met: list[str] = Field(
        default_factory=list,
        description="Policy criteria that the submitted clinical evidence "
                    "satisfies. Each entry should map to a specific criterion "
                    "in the coverage policy.",
        examples=[["Radiographic evidence of severe OA (KL grade ≥3)",
                   "Failed ≥3 months of conservative therapy"]],
    )
    """Audit trail and UI display for the nurse; shows what the AI found
    to be sufficient in the submitted evidence."""

    criteria_not_met: list[str] = Field(
        default_factory=list,
        description="Policy criteria that are not satisfied or not documented "
                    "in the submitted evidence. Populated for DENY or "
                    "INSUFFICIENT_INFO decisions.",
        examples=[["No documentation of trial with viscosupplementation"]],
    )
    """Drives denial-reason letter generation and informs nurse overrides."""

    @model_validator(mode="after")
    def validate_insufficient_info_has_missing(self) -> "ClinicalReasoningOutput":
        """Enforce internal consistency: INSUFFICIENT_INFO requires at least
        one entry in missing_information so the provider knows what to supply."""
        if (
            self.decision == AIDecision.INSUFFICIENT_INFO
            and not self.missing_information
        ):
            raise ValueError(
                "missing_information must contain at least one item when "
                "decision is INSUFFICIENT_INFO."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision": "APPROVE",
                "confidence_score": 0.91,
                "policy_basis": "Coverage Policy CP-2024-0041 §3.2",
                "reasoning": (
                    "The clinical notes document KL grade 4 bilateral OA and "
                    "six months of failed conservative therapy including PT, "
                    "NSAIDs, and cortisone injections, meeting all required "
                    "criteria for TKA under CP-2024-0041 §3.2."
                ),
                "missing_information": [],
                "criteria_met": [
                    "Radiographic evidence of severe OA (KL grade ≥3)",
                    "Failed ≥3 months of conservative therapy",
                    "Conservative therapy included at least two modalities",
                ],
                "criteria_not_met": [],
            }
        }
    }


# ---------------------------------------------------------------------------
# 3. NurseDecision — human-in-the-loop review input
# ---------------------------------------------------------------------------


class NurseDecision(BaseModel):
    """Input recorded when a licensed nurse reviews and finalises a case.

    This schema is accepted by the POST /cases/{case_id}/review endpoint.
    The nurse either confirms the AI recommendation or overrides it; in either
    case a reason must be provided when overriding.
    """

    decision: NurseDecisionEnum = Field(
        description="The nurse's final disposition: APPROVED, DENIED, or "
                    "REQUEST_MORE_INFO.",
    )
    """Written to the case record as the authoritative final decision.
    Supersedes the AI recommendation for regulatory and audit purposes."""

    nurse_id: Annotated[str, Field(
        min_length=1,
        max_length=50,
        description="Employee or credential ID of the reviewing nurse. Used "
                    "for audit logging and regulatory compliance (CMS requires "
                    "human review decisions to be attributed to a licensed "
                    "clinician).",
        examples=["RN-78901"],
    )]
    """Required for HIPAA audit trail; every state change on a case must be
    attributable to a specific, credentialed individual."""

    override_reason: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Required when the nurse's decision differs from the AI "
                    "recommendation. Documents the clinical or policy rationale "
                    "for the override.",
        examples=["Patient is immunocompromised; additional conservative "
                  "therapy would pose unacceptable risk."],
    )
    """Mandatory for overrides to ensure decisions can withstand appeals and
    regulatory audits. Validated by model_validator below."""

    notes: Optional[str] = Field(
        default=None,
        max_length=5_000,
        description="Free-text additional observations from the nurse. Not "
                    "required but surfaced in the case audit log.",
    )
    """Informal notes; not used in downstream logic but preserved for
    completeness in the audit record."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "decision": "APPROVED",
                "nurse_id": "RN-78901",
                "override_reason": None,
                "notes": "Confirmed AI recommendation; documentation complete.",
            }
        }
    }


# ---------------------------------------------------------------------------
# 4. CaseStatusResponse — status polling API response
# ---------------------------------------------------------------------------


class CaseStatusResponse(BaseModel):
    """Response body returned by GET /cases/{case_id}/status.

    Exposes the current lifecycle state, the AI recommendation, and the nurse
    decision (if one has been recorded). Intended for EHR polling integrations
    and the provider portal.
    """

    case_id: Annotated[str, Field(
        description="UUID of the prior-auth case, assigned at submission.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )]

    status: CaseStatus = Field(
        description="Current lifecycle state of the case.",
    )
    """Drives UI state in the provider portal and determines whether the case
    is actionable (e.g., provider can submit more info)."""

    member_id: Annotated[str, Field(
        description="Health-plan member identifier, echoed from the original "
                    "request for correlation.",
        examples=["MBR-00123456"],
    )]

    ai_recommendation: Optional[AIDecision] = Field(
        default=None,
        description="The AI's recommended decision. None until the "
                    "clinical-reasoning node has completed.",
    )
    """Shown to nurses in the review queue. None during PENDING_AI_REVIEW."""

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence associated with ai_recommendation. "
                    "None until AI processing completes.",
    )
    """Displayed alongside the AI recommendation to help nurses prioritise
    which cases need closest scrutiny."""

    requires_human_review: bool = Field(
        description="True when the case needs a nurse to review before a "
                    "final decision is issued. Set automatically based on "
                    "confidence_score threshold and decision type.",
    )
    """Drives nurse work-queue inclusion; also exposed to providers so they
    know to expect a human-reviewed turnaround time."""

    submitted_at: datetime = Field(
        description="UTC timestamp of the original submission.",
    )
    """Used to calculate SLA compliance (e.g., 72-hour urgent deadline)."""

    nurse_decision: Optional[NurseDecision] = Field(
        default=None,
        description="The nurse's recorded decision, or None if human review "
                    "has not yet occurred.",
    )
    """None until a nurse submits a review via the /review endpoint."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "case_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "PENDING_NURSE_REVIEW",
                "member_id": "MBR-00123456",
                "ai_recommendation": "APPROVE",
                "confidence_score": 0.91,
                "requires_human_review": True,
                "submitted_at": "2026-03-25T10:30:00Z",
                "nurse_decision": None,
            }
        }
    }


# ---------------------------------------------------------------------------
# 5. PriorAuthResponse — submission acknowledgement API response
# ---------------------------------------------------------------------------


class PriorAuthResponse(BaseModel):
    """Response body returned immediately after a successful POST /cases request.

    This is an acknowledgement, not a decision. It gives the provider the
    case ID they need to poll status and sets turnaround expectations.
    """

    case_id: Annotated[str, Field(
        description="UUID assigned to the newly created case. "
                    "Store this — it is required for all subsequent calls.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )]
    """Primary key for the case in the database; returned immediately so
    providers can begin polling without a separate lookup."""

    status: CaseStatus = Field(
        default=CaseStatus.PENDING_AI_REVIEW,
        description="Initial case status. Always PENDING_AI_REVIEW at "
                    "submission time.",
    )
    """Confirms the case has entered the processing pipeline."""

    message: str = Field(
        description="Human-readable acknowledgement message suitable for "
                    "display in the provider portal or EHR UI.",
        examples=["Prior authorization request received and queued for "
                  "AI-assisted clinical review."],
    )
    """Gives the submitting provider clear, plain-English confirmation that
    the request was accepted and what happens next."""

    estimated_decision_time: str = Field(
        description="Plain-English estimate of when a decision will be issued. "
                    "Derived from is_urgent and current queue depth. "
                    "Not a binding SLA commitment.",
        examples=["Within 2 business days", "Within 72 hours (urgent)"],
    )
    """Regulatory requirement: CMS mandates that providers receive a turnaround
    estimate at submission time for expedited and standard requests."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "case_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "PENDING_AI_REVIEW",
                "message": (
                    "Prior authorization request received and queued for "
                    "AI-assisted clinical review."
                ),
                "estimated_decision_time": "Within 2 business days",
            }
        }
    }
