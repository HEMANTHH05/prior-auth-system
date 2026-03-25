"""
app/services/rag_service.py

Policy Retrieval-Augmented Generation (RAG) service for the Prior Authorization
AI system.

Called by Node 4 (policy_retrieval_node) in the LangGraph workflow to fetch
the policy document chunks most relevant to a prior-auth request *before* the
clinical-reasoning LLM node runs.  The retrieved chunks are injected into the
LLM's context window as authoritative coverage-policy grounding.

Architecture:
  Hybrid search = keyword BM25 (text search) + semantic vector search
  ──────────────────────────────────────────────────────────────────
  Azure AI Search supports "hybrid search" natively: a single API call that
  runs both a full-text BM25 query and a vector k-NN query, then fuses their
  rankings via Reciprocal Rank Fusion (RRF).  This outperforms either alone:
  - BM25 catches exact CPT/ICD code matches that embeddings may miss.
  - Vector search catches semantically related criteria even when the exact
    codes aren't in the document.

Filter strategy:
  We always want documents that apply to ALL plans ("all") plus documents
  specific to the member's insurance plan.  We then narrow by procedure_type
  when we can confidently detect it, so the LLM doesn't receive irrelevant
  policy sections (e.g., mental-health criteria for a cardiac request).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read once at import, fail loudly if missing
# ---------------------------------------------------------------------------

_SEARCH_ENDPOINT: str = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
_SEARCH_KEY: str = os.environ.get("AZURE_SEARCH_KEY", "")
_SEARCH_INDEX: str = os.environ.get("AZURE_SEARCH_INDEX", "prior-auth-policies")
_OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# text-embedding-3-small produces 1536-dimensional vectors, matching the index.
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIMENSIONS = 1536

for _var_name, _var_val in [
    ("AZURE_SEARCH_ENDPOINT", _SEARCH_ENDPOINT),
    ("AZURE_SEARCH_KEY", _SEARCH_KEY),
    ("OPENAI_API_KEY", _OPENAI_API_KEY),
]:
    if not _var_val:
        logger.warning("%s is not set. RAG searches will fail.", _var_name)

# ---------------------------------------------------------------------------
# Lazy-initialised SDK clients
# ---------------------------------------------------------------------------
# We initialise clients once and reuse them across calls in the same process.
# This avoids the overhead of creating an HTTP session per LangGraph node
# invocation (which can add ~200 ms per cold-start on Azure).

_search_client: SearchClient | None = None
_openai_client: OpenAI | None = None


def _get_search_client() -> SearchClient:
    """Return (or create) the module-level Azure AI Search client."""
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            endpoint=_SEARCH_ENDPOINT,
            index_name=_SEARCH_INDEX,
            credential=AzureKeyCredential(_SEARCH_KEY),
        )
    return _search_client


def _get_openai_client() -> OpenAI:
    """Return (or create) the module-level OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=_OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Plan name mapping
# ---------------------------------------------------------------------------

# Insurance plan names from our Supabase members data (as stored in
# insurance_plans.name) mapped to the discrete filter values used in the
# Azure AI Search index (plan_name field).
#
# "all" documents contain universal coverage rules that apply to every plan
# (e.g., CMS requirements, clinical practice guidelines).  Plan-specific
# documents contain plan-specific riders, exclusions, or enhanced benefits.
_PLAN_NAME_MAP: dict[str, str] = {
    # Commercial / private insurers
    "aetna": "private",
    "anthem": "private",
    "blue cross blue shield": "private",
    "bcbs": "private",
    "cigna health": "private",
    "cigna": "private",
    "humana": "private",
    "unitedhealthcare": "private",
    "united health": "private",
    "united healthcare": "private",
    # Government programs
    "medicare": "medicare",
    "dual eligible": "medicare",   # dual-eligible members follow Medicare rules
    "medicaid": "medicaid",
}


def map_plan_name(db_plan_name: str) -> str:
    """Map an insurance_plans.name value to the Azure index plan_name filter.

    Normalises the input to lowercase before lookup so "Aetna", "AETNA",
    and "aetna" all resolve correctly.

    Args:
        db_plan_name: The plan name string as stored in the insurance_plans
                      table (e.g., "UnitedHealthcare").

    Returns:
        One of "private", "medicare", "medicaid".
        Falls back to "private" if the plan is unrecognised — commercial
        private-plan documents are the most broadly applicable fallback.
    """
    normalised = db_plan_name.strip().lower()
    mapped = _PLAN_NAME_MAP.get(normalised)
    if mapped is None:
        logger.warning(
            "Unrecognised plan name '%s'; defaulting to 'private'.", db_plan_name
        )
        return "private"
    return mapped


# ---------------------------------------------------------------------------
# CPT → procedure type detection
# ---------------------------------------------------------------------------

# CPT code ranges that map to each procedure_type value in the index.
# Stored as (inclusive_start, inclusive_end, procedure_type) tuples.
# Ordered from most specific to least specific so the first match wins.
#
# Sources: AMA CPT code book section headers, CMS claims processing manual.
_CPT_RANGES: list[tuple[int, int, str]] = [
    # MRI — diagnostic imaging section (radiology)
    (70540, 70559, "mri"),   # MRI head and neck
    (72141, 72158, "mri"),   # MRI spine (cervical, thoracic, lumbar)
    (73218, 73223, "mri"),   # MRI upper extremity
    (73721, 73725, "mri"),   # MRI lower extremity
    (74181, 74183, "mri"),   # MRI abdomen
    (75557, 75564, "mri"),   # Cardiac MRI
    # Cardiac — cardiovascular section
    (93000, 93799, "cardiac"),
    # Mental health — psychiatry section
    (90785, 90899, "mental_health"),
    # Physical therapy — physical medicine & rehabilitation
    (97010, 97799, "physical_therapy"),
    # Specialist referral — evaluation & management office/outpatient
    (99202, 99215, "specialist_referral"),
    (99241, 99255, "specialist_referral"),  # consultations
    # Diabetes management (overlaps E&M; listed separately for specificity)
    (95250, 95251, "diabetes"),   # CGM training
]

# Keyword signals in the procedure description — checked when CPT range
# lookup is inconclusive.  Checked in order; first match wins.
_DESCRIPTION_KEYWORDS: list[tuple[str, str]] = [
    ("mri", "mri"),
    ("magnetic resonance", "mri"),
    ("cardiac", "cardiac"),
    ("cardio", "cardiac"),
    ("echocardiogram", "cardiac"),
    ("heart", "cardiac"),
    ("ekg", "cardiac"),
    ("ecg", "cardiac"),
    ("physical therapy", "physical_therapy"),
    ("physiotherapy", "physical_therapy"),
    ("therapeutic exercise", "physical_therapy"),
    ("psychiatr", "mental_health"),
    ("psychotherapy", "mental_health"),
    ("mental health", "mental_health"),
    ("behavioral health", "mental_health"),
    ("diabetes", "diabetes"),
    ("glucose", "diabetes"),
    ("insulin", "diabetes"),
    ("surgery", "surgery"),
    ("surgical", "surgery"),
    ("arthroplasty", "surgery"),
    ("resection", "surgery"),
    ("colectomy", "surgery"),
    ("mastectomy", "surgery"),
    ("referral", "specialist_referral"),
    ("consultation", "specialist_referral"),
]


def detect_procedure_type(cpt_code: str, description: str = "") -> str | None:
    """Infer the Azure index procedure_type value for a CPT code.

    Two-pass detection:
    1. Try CPT numeric range lookup (most reliable — no NLP needed).
    2. Fall back to keyword scan of the procedure description.

    Returns None when no confident match is found so the caller can omit
    the procedure_type filter entirely rather than over-restricting results.

    Args:
        cpt_code:    5-digit CPT code string (e.g., "72148").
        description: Human-readable procedure name (e.g., "MRI lumbar spine
                     without contrast").  Optional but improves detection.

    Returns:
        One of the procedure_type index values or None.
    """
    # --- Pass 1: numeric CPT range ---
    try:
        code_int = int(cpt_code.strip())
        for start, end, ptype in _CPT_RANGES:
            if start <= code_int <= end:
                logger.debug(
                    "CPT %s matched range %d-%d → procedure_type=%s",
                    cpt_code, start, end, ptype,
                )
                return ptype
    except ValueError:
        # Non-numeric CPT (e.g., category-II or III codes starting with letters)
        logger.debug("CPT '%s' is non-numeric; skipping range lookup.", cpt_code)

    # --- Pass 2: keyword scan of description ---
    desc_lower = description.lower()
    for keyword, ptype in _DESCRIPTION_KEYWORDS:
        if keyword in desc_lower:
            logger.debug(
                "Description keyword '%s' matched → procedure_type=%s",
                keyword, ptype,
            )
            return ptype

    logger.debug(
        "No procedure_type detected for CPT='%s' description='%s'; "
        "will omit procedure_type filter.",
        cpt_code, description,
    )
    return None


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def _get_embedding(text: str) -> list[float]:
    """Generate a 1536-dimension embedding for the given text using OpenAI.

    Why text-embedding-3-small?
    - Natively produces 1536-dimensional vectors, matching our index schema
      without truncation.
    - ~5× cheaper than text-embedding-ada-002 with comparable retrieval
      quality on domain-specific text (Azure benchmarks, 2024).
    - Latency is low enough for synchronous calls within a LangGraph node.

    Args:
        text: Query string to embed (CPT + ICD + clinical context).

    Returns:
        List of 1536 floats.

    Raises:
        Exception: Propagated to caller; handled in search_policies().
    """
    client = _get_openai_client()
    response = client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=text,
        dimensions=_EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# OData filter builder
# ---------------------------------------------------------------------------


def _build_filter(mapped_plan: str, procedure_type: str | None) -> str:
    """Construct the OData $filter expression for the Azure AI Search query.

    Filter logic:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Always include:  plan_name eq 'all'                                │
    │  Also include:    plan_name eq '<mapped_plan>'   (if not 'all')     │
    │  Optionally add:  procedure_type eq '<procedure_type>'              │
    └─────────────────────────────────────────────────────────────────────┘

    "all" plan documents contain universal medical-necessity criteria that
    apply regardless of plan type.  We must always include them or the LLM
    may lack the foundational policy text it needs to make a decision.

    We add the procedure_type filter only when we're confident of the type
    (detect_procedure_type returned non-None).  Omitting it is safer than
    filtering too aggressively and missing relevant policy chunks.

    Args:
        mapped_plan:    Index plan_name value: "private", "medicare", "medicaid".
        procedure_type: Index procedure_type value or None.

    Returns:
        OData filter string ready to pass to SearchClient.search().
    """
    # Plan clause: always include "all" + the member's specific plan
    if mapped_plan == "all":
        plan_filter = "plan_name eq 'all'"
    else:
        plan_filter = f"(plan_name eq 'all' or plan_name eq '{mapped_plan}')"

    # Procedure type clause: only added when detection is confident
    if procedure_type:
        return f"{plan_filter} and procedure_type eq '{procedure_type}'"

    return plan_filter


# ---------------------------------------------------------------------------
# Main search function — called by policy_retrieval_node
# ---------------------------------------------------------------------------


def search_policies(
    cpt_code: str,
    icd_code: str,
    plan_name: str,
    procedure_description: str = "",
    top: int = 5,
) -> list[dict[str, Any]]:
    """Search the Azure AI Search policy index for relevant coverage criteria.

    Executes a hybrid search (BM25 + vector) so the clinical-reasoning node
    receives the most relevant policy chunks for this specific prior-auth
    request.  Results are ranked by Azure's Reciprocal Rank Fusion (RRF)
    algorithm, which fuses BM25 and vector scores.

    Args:
        cpt_code:              5-digit CPT code for the requested procedure.
        icd_code:              ICD-10-CM diagnosis code.
        plan_name:             Insurance plan name as stored in Supabase
                               (e.g., "UnitedHealthcare").
        procedure_description: Human-readable procedure name.  Used to improve
                               procedure type detection and the search query.
        top:                   Number of policy chunks to retrieve (default 5).
                               5 chunks keeps the LLM context manageable while
                               providing adequate policy coverage.

    Returns:
        List of dicts, each representing one policy document chunk::

            {
                "id":             str,   # document key in the index
                "content":        str,   # the policy text chunk
                "filename":       str,   # source PDF filename
                "plan_name":      str,   # "all" | "private" | "medicare" | ...
                "procedure_type": str,   # "mri" | "surgery" | ...
                "score":          float, # RRF relevance score (higher = better)
            }

        Returns [] on any error so the LangGraph node can continue with
        degraded context rather than crashing the workflow.
    """
    # --- Step 1: build the search query text ---
    # Combine CPT, ICD, and description into a single query string.
    # BM25 will match exact code strings; the description adds semantic signal.
    query_text = f"CPT {cpt_code} ICD {icd_code} {procedure_description}".strip()

    # --- Step 2: map plan name and detect procedure type ---
    mapped_plan = map_plan_name(plan_name)
    procedure_type = detect_procedure_type(cpt_code, procedure_description)

    # --- Step 3: build OData filter ---
    odata_filter = _build_filter(mapped_plan, procedure_type)

    logger.info(
        "RAG search | CPT=%s ICD=%s plan=%s→%s proc_type=%s filter=%r",
        cpt_code, icd_code, plan_name, mapped_plan, procedure_type, odata_filter,
    )

    # --- Step 4: generate query embedding ---
    try:
        embedding = _get_embedding(query_text)
    except Exception as exc:
        logger.error("Embedding generation failed: %s — falling back to text-only search.", exc)
        embedding = None

    # --- Step 5: execute hybrid search ---
    try:
        client = _get_search_client()

        # VectorizedQuery wraps a pre-computed embedding for k-NN search.
        # exhaustive=False uses the HNSW approximate index (faster, ~99% recall).
        vector_queries = []
        if embedding is not None:
            vector_queries = [
                VectorizedQuery(
                    vector=embedding,
                    k_nearest_neighbors=top,
                    fields="content_vector",
                    exhaustive=False,
                )
            ]

        raw_results = client.search(
            search_text=query_text,         # BM25 full-text query
            vector_queries=vector_queries,  # vector k-NN query (empty = text-only)
            filter=odata_filter,
            select=["id", "content", "filename", "plan_name", "procedure_type"],
            top=top,
        )

        results: list[dict[str, Any]] = []
        for doc in raw_results:
            results.append({
                "id": doc.get("id", ""),
                "content": doc.get("content", ""),
                "filename": doc.get("filename", ""),
                "plan_name": doc.get("plan_name", ""),
                "procedure_type": doc.get("procedure_type", ""),
                # @search.score is the RRF-fused relevance score
                "score": doc.get("@search.score", 0.0),
            })

        logger.info(
            "RAG search returned %d chunks (top=%d) | query=%r",
            len(results), top, query_text,
        )
        return results

    except Exception as exc:
        logger.error("Azure AI Search query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Test / smoke-test function
# ---------------------------------------------------------------------------


def test_rag_search() -> None:
    """Run a smoke-test search for 'MRI lumbar spine' and print results.

    Intended for manual verification after indexing new documents or
    rotating credentials.  Not part of the LangGraph workflow.

    Usage::

        source venv/bin/activate
        python -m app.services.rag_service
    """
    import json

    print("=" * 70)
    print("RAG smoke test — searching for 'MRI lumbar spine'")
    print("=" * 70)

    # Simulate a typical prior-auth for lumbar MRI from a private-plan member
    results = search_policies(
        cpt_code="72148",               # MRI lumbar spine without contrast
        icd_code="M54.5",               # Low back pain
        plan_name="UnitedHealthcare",   # → maps to "private"
        procedure_description="MRI lumbar spine without contrast",
        top=5,
    )

    if not results:
        print("\n[!] No results returned. Check credentials and index content.")
        return

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"  id:             {doc['id']}")
        print(f"  filename:       {doc['filename']}")
        print(f"  plan_name:      {doc['plan_name']}")
        print(f"  procedure_type: {doc['procedure_type']}")
        print(f"  score:          {doc['score']:.4f}")
        # Truncate long content for readability in the terminal
        content_preview = doc["content"][:400].replace("\n", " ")
        print(f"  content:        {content_preview}...")

    print("\n" + "=" * 70)
    print(f"Total results: {len(results)}")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)
    test_rag_search()
