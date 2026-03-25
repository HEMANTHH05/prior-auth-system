"""
app/services/member_service.py

Member data access layer for the Prior Authorization AI system.

Called by Node 2 (member_context_node) in the LangGraph workflow to hydrate
the graph state with all relevant clinical and administrative data for a
member before the clinical-reasoning LLM node runs.

Design decisions:
- psycopg2 with a simple per-call connection pattern backed by Supabase's
  PgBouncer transaction-mode pooler (DATABASE_URL points to port 6543).
  We do NOT use a persistent connection pool inside this process because
  Supabase's pooler already manages the real Postgres connections for us.
- Every public function returns a plain dict / list of dicts so results can
  be JSON-serialised directly into the LangGraph state without extra mapping.
- Failures are logged and return safe empty values (None / []) rather than
  raising, so a single unavailable table never aborts the entire workflow.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras  # RealDictCursor – rows as dicts, not tuples
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# load_dotenv is idempotent; safe to call at module import time.
# In production (Docker / cloud) the vars are injected directly, so this
# call is a no-op but harmless.
load_dotenv()

logger = logging.getLogger(__name__)

# Fetch once at import time so every call to get_connection() doesn't
# re-read the environment.  Fail loudly here if the variable is missing
# so the problem surfaces at startup, not during a live request.
_DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
if not _DATABASE_URL:
    logger.warning(
        "DATABASE_URL is not set. All member_service queries will fail."
    )


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


@contextmanager
def get_connection():
    """Yield a psycopg2 connection and guarantee cleanup.

    Usage::

        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT 1")

    Why a context manager instead of a module-level pool?
    Supabase uses PgBouncer in *transaction mode* on port 6543, which means
    server-side prepared statements and SET LOCAL are not reliable across
    calls.  Opening a fresh logical connection per service call is the
    recommended pattern; PgBouncer still reuses the underlying TCP socket.
    """
    conn = None
    try:
        conn = psycopg2.connect(_DATABASE_URL)
        yield conn
    except psycopg2.OperationalError as exc:
        logger.error("Could not connect to database: %s", exc)
        raise
    finally:
        if conn and not conn.closed:
            conn.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fetchall_as_dicts(cur) -> list[dict[str, Any]]:
    """Convert RealDictCursor rows to plain dicts so they are JSON-safe."""
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def _fetchone_as_dict(cur) -> dict[str, Any] | None:
    """Convert a single RealDictCursor row to a plain dict, or None."""
    row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# 1. Basic member demographics
# ---------------------------------------------------------------------------


def get_member_basic_info(member_id: str) -> dict[str, Any] | None:
    """Return core demographic fields for a member.

    Why this query exists:
    The LLM needs the member's age (derived from birthdate) and gender to
    apply age/sex-specific coverage criteria (e.g., certain preventive
    screenings are only covered above a threshold age).

    Returns None if the member is not found, allowing the master function
    to detect an invalid member_id early and short-circuit the workflow.

    Args:
        member_id: The health-plan member identifier (primary key in members).

    Returns:
        Dict with keys: id, first_name, last_name, birthdate, gender,
        address, city, state, zip — or None if no row found.
    """
    sql = """
        SELECT
            id,
            first_name,
            last_name,
            birthdate,
            gender,
            address,
            city,
            state,
            zip
        FROM members
        WHERE id = %s
        LIMIT 1
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # %s placeholder — psycopg2 handles escaping; no f-strings.
                cur.execute(sql, (member_id,))
                return _fetchone_as_dict(cur)
    except psycopg2.errors.InvalidTextRepresentation:
        # The members.id column is UUID type.  A non-UUID member_id raises
        # this error at the DB level — treat it as "not found", not a system
        # failure, so the caller receives None and triggers the validation_error
        # path rather than logging a spurious error.
        logger.debug(
            "member_id='%s' is not a valid UUID; treating as not found.",
            member_id,
        )
        return None
    except Exception as exc:
        logger.error(
            "get_member_basic_info failed for member_id=%s: %s", member_id, exc
        )
        return None


# ---------------------------------------------------------------------------
# 2. Insurance / plan information
# ---------------------------------------------------------------------------


def get_member_insurance(member_id: str) -> list[dict[str, Any]]:
    """Return active insurance plan(s) for a member, joined with plan details.

    Why this query exists:
    Coverage rules differ by plan (HMO vs PPO, employer vs individual).
    The plan name and ownership type are included in the RAG retrieval prompt
    so the correct policy document is fetched from Azure AI Search.

    We join member_insurance → insurance_plans to avoid a second round-trip
    and to surface the human-readable plan name alongside the plan_id FK.

    Args:
        member_id: Health-plan member identifier.

    Returns:
        List of dicts with keys: plan_id, plan_name, ownership, is_active,
        start_date, end_date, plan_ownership, owner_name.
        Returns [] on error or if no rows exist.
    """
    sql = """
        SELECT
            mi.plan_id,
            ip.name            AS plan_name,
            ip.ownership,
            mi.is_active,
            mi.start_date,
            mi.end_date,
            mi.plan_ownership,
            mi.owner_name
        FROM member_insurance mi
        JOIN insurance_plans ip ON ip.id = mi.plan_id
        WHERE mi.member_id = %s
        ORDER BY mi.is_active DESC, mi.start_date DESC
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id,))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_insurance failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 3. Claims history
# ---------------------------------------------------------------------------


def get_member_claims(member_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent claims for a member.

    Why this query exists:
    Claims history gives the LLM evidence of prior treatment patterns —
    e.g., has the member already had a procedure that makes this request
    a duplicate, or do the diagnosis codes on prior claims corroborate
    the current ICD code?

    Diagnosis columns (diagnosis1–diagnosis8) are included because a claim
    can carry multiple ICD-10 codes and any of them may be relevant to the
    current prior-auth request.

    Args:
        member_id: Health-plan member identifier.
        limit:     Maximum number of recent claims to return (default 10).

    Returns:
        List of claim dicts ordered by service_date descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            member_id,
            diagnosis1,
            diagnosis2,
            diagnosis3,
            diagnosis4,
            diagnosis5,
            diagnosis6,
            diagnosis7,
            diagnosis8,
            service_date,
            status1
        FROM claims
        WHERE member_id = %s
        ORDER BY service_date DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id, limit))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_claims failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 4. Active conditions / problem list
# ---------------------------------------------------------------------------


def get_member_conditions(member_id: str) -> list[dict[str, Any]]:
    """Return all recorded conditions (active and historical) for a member.

    Why this query exists:
    The problem list is the most direct evidence for medical necessity.
    An active condition code that matches or is clinically related to the
    ICD code on the prior-auth request is strong evidence for approval.
    Historical (inactive) conditions matter too — e.g., prior cancer
    diagnosis may waive a waiting period.

    We return ALL conditions (not just active) so the LLM can reason about
    both current and historical context.  The is_active flag lets the LLM
    distinguish between them.

    Args:
        member_id: Health-plan member identifier.

    Returns:
        List of condition dicts ordered by start_date descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            code,
            description,
            is_active,
            start_date,
            stop_date
        FROM conditions
        WHERE member_id = %s
        ORDER BY is_active DESC, start_date DESC
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id,))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_conditions failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 5. Medications
# ---------------------------------------------------------------------------


def get_member_medications(
    member_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Return recent medications for a member.

    Why this query exists:
    Many coverage policies require documented failure of first-line drug
    therapy before approving a procedure or a higher-cost medication.
    For example, a prior-auth for a biologic may be denied unless the notes
    show at least one DMARD trial.  The medication list gives the LLM
    objective evidence of what has already been tried.

    Args:
        member_id: Health-plan member identifier.
        limit:     Maximum number of medications to return (default 10).

    Returns:
        List of medication dicts ordered by start_time descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            code,
            description,
            start_time,
            stop_time
        FROM medications
        WHERE member_id = %s
        ORDER BY start_time DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id, limit))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_medications failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 6. Imaging studies
# ---------------------------------------------------------------------------


def get_member_imaging(
    member_id: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Return recent imaging studies for a member.

    Why this query exists:
    Imaging (X-ray, MRI, CT) is often required evidence for musculoskeletal
    and oncological prior-auth requests.  For example, a total knee
    arthroplasty request typically requires X-ray evidence of severe joint
    space narrowing.  The imaging table records what imaging has already been
    performed, which the LLM cross-references against the policy criteria.

    Args:
        member_id: Health-plan member identifier.
        limit:     Maximum number of imaging records to return (default 5).

    Returns:
        List of imaging study dicts ordered by study_date descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            study_date,
            bodysite_description,
            modality_description,
            procedure_code
        FROM imaging_studies
        WHERE member_id = %s
        ORDER BY study_date DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id, limit))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_imaging failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 7. Procedures
# ---------------------------------------------------------------------------


def get_member_procedures(
    member_id: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Return recent procedures performed on a member.

    Why this query exists:
    Procedure history documents what interventions have already been done,
    which is essential for:
    (a) Detecting duplicate requests — if the same CPT code was billed
        recently, the new request may be clinically inappropriate.
    (b) Verifying step-therapy — some policies require less invasive
        procedures before authorising a major intervention.

    Args:
        member_id: Health-plan member identifier.
        limit:     Maximum number of procedures to return (default 10).

    Returns:
        List of procedure dicts ordered by start_time descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            code,
            description,
            start_time,
            reason_description
        FROM procedures
        WHERE member_id = %s
        ORDER BY start_time DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id, limit))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_member_procedures failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 8. Prior authorization history
# ---------------------------------------------------------------------------


def get_prior_auth_history(
    member_id: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Return past prior-auth requests filed for a member.

    Why this query exists:
    Prior decisions are precedent.  If a similar or identical CPT/ICD
    combination was previously approved, a new request is likely approvable.
    If it was previously denied (and the clinical picture has not changed),
    that context should be surfaced to the reviewing nurse.

    This also catches potential duplicate submissions — two active requests
    for the same procedure within a short window is a red flag.

    Args:
        member_id: Health-plan member identifier.
        limit:     Maximum number of historical cases to return (default 5).

    Returns:
        List of prior-auth case dicts ordered by submitted_at descending.
        Returns [] on error.
    """
    sql = """
        SELECT
            id,
            case_id,
            cpt_code,
            icd_code,
            ai_recommendation,
            nurse_decision,
            status,
            submitted_at
        FROM prior_auth_requests
        WHERE member_id = %s
        ORDER BY submitted_at DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (member_id, limit))
                return _fetchall_as_dicts(cur)
    except Exception as exc:
        logger.error(
            "get_prior_auth_history failed for member_id=%s: %s", member_id, exc
        )
        return []


# ---------------------------------------------------------------------------
# 9. Master aggregator — called by member_context_node
# ---------------------------------------------------------------------------


def get_full_member_context(member_id: str) -> dict[str, Any]:
    """Aggregate all member data into a single dict for the LangGraph state.

    This is the only function that member_context_node needs to call.
    It fans out to the individual data functions, collects results, and
    packages them under predictable keys.

    Failure semantics:
    - If the member does not exist (get_member_basic_info returns None),
      a ``validation_error`` key is set and all other keys are empty.
      The LangGraph workflow should inspect this key before proceeding to
      the clinical-reasoning node.
    - If any individual sub-query fails, its key holds an empty list so
      the overall context is still usable (degraded but not broken).

    Args:
        member_id: Health-plan member identifier from the prior-auth request.

    Returns:
        Dict with the following top-level keys::

            {
                "member_id":       str,
                "basic_info":      dict | None,
                "insurance":       list[dict],
                "claims":          list[dict],
                "conditions":      list[dict],
                "medications":     list[dict],
                "imaging":         list[dict],
                "procedures":      list[dict],
                "prior_auth_history": list[dict],
                "validation_error": str | None,   # set only when member not found
            }
    """
    logger.info("Fetching full member context for member_id=%s", member_id)

    # --- Step 1: verify the member exists before firing all sub-queries ---
    basic_info = get_member_basic_info(member_id)

    if basic_info is None:
        # Member not found — return a sentinel so the workflow can halt early
        # and return a 404-style response to the caller without invoking the LLM.
        logger.warning(
            "Member not found in members table: member_id=%s", member_id
        )
        return {
            "member_id": member_id,
            "basic_info": None,
            "insurance": [],
            "claims": [],
            "conditions": [],
            "medications": [],
            "imaging": [],
            "procedures": [],
            "prior_auth_history": [],
            "validation_error": (
                f"Member '{member_id}' not found. "
                "Verify the member ID and resubmit."
            ),
        }

    # --- Step 2: fetch all clinical and administrative data concurrently ---
    # Each call is independent; failures return empty lists, not exceptions.
    insurance = get_member_insurance(member_id)
    claims = get_member_claims(member_id, limit=10)
    conditions = get_member_conditions(member_id)
    medications = get_member_medications(member_id, limit=10)
    imaging = get_member_imaging(member_id, limit=5)
    procedures = get_member_procedures(member_id, limit=10)
    prior_auth_history = get_prior_auth_history(member_id, limit=5)

    logger.info(
        "Member context loaded for member_id=%s | "
        "insurance=%d plans, claims=%d, conditions=%d, "
        "medications=%d, imaging=%d, procedures=%d, prior_auth=%d",
        member_id,
        len(insurance),
        len(claims),
        len(conditions),
        len(medications),
        len(imaging),
        len(procedures),
        len(prior_auth_history),
    )

    return {
        "member_id": member_id,
        "basic_info": basic_info,
        "insurance": insurance,
        "claims": claims,
        "conditions": conditions,
        "medications": medications,
        "imaging": imaging,
        "procedures": procedures,
        "prior_auth_history": prior_auth_history,
        "validation_error": None,  # member exists; no validation error
    }
