# Prior Authorization AI System

A full-stack AI system that automates prior authorization decisions for a health insurance plan using LangGraph, GPT-4o, Azure AI Search, and Supabase — built as a personal portfolio project to demonstrate agentic AI workflows in healthcare.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple) ![FastAPI](https://img.shields.io/badge/FastAPI-0.11-green) ![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-orange)

---

## Overview

Prior authorization (PA) is one of the most administratively burdensome parts of U.S. healthcare. Physicians must submit clinical justification for procedures, and reviewers must match that justification against insurer policy — a process that often takes 3–7 business days and is largely manual.

This project replaces that manual workflow with an AI-assisted pipeline:

1. A physician submits a PA request via API (member ID, CPT code, ICD-10 code, clinical notes)
2. An 8-node LangGraph workflow classifies the request, fetches member context, retrieves relevant policy chunks from a vector search index, and runs GPT-4o clinical reasoning
3. High-confidence cases are auto-approved; low-confidence or DENY cases are paused for a nurse to review via a web portal
4. Every case is written to Supabase with a full audit trail

---

## Architecture

```
POST /prior-auth/submit
         │
         ▼
[intent_classifier]        Classifies: prior_auth / eligibility / member_service / unknown
         │ route_after_intent
         ▼
[member_context]           Fetches full member record from Supabase (8 sub-queries)
         │ route_after_member_context
         ├─(attachments)─► [document_ingestion]  PyMuPDF PDF extraction
         │                         │
         ▼                         ▼
[policy_retrieval]         Azure AI Search hybrid retrieval (BM25 + vector HNSW)
         │
         ▼
[clinical_reasoning]       GPT-4o: APPROVE / DENY / INSUFFICIENT_INFO + confidence score
         │
         ▼
[confidence_evaluation]    Threshold logic: confidence < 0.75, DENY, urgent → human review
         │ route_after_confidence
         ├─(needs review)─► [human_review]  ◄── INTERRUPT: paused until nurse submits
         │                         │
         ▼                         ▼
[audit_output]             Writes case to Supabase, assembles API response
         │
         ▼
        END


Error / non-PA shortcuts:
  intent_classifier ──► [error_output] ──► END    (unknown intent)
  member_context    ──► [error_output] ──► END    (member not found)
  member_context    ──► [audit_output] ──► END    (eligibility / member_service)
```

---

## Two Data Streams

The graph combines two independent data sources before clinical reasoning runs:

**Stream 1 — Member context (Supabase)**
`member_context_node` runs 8 parallel SQL queries to build a complete member record:
basic demographics, insurance plan, recent claims, active diagnoses, current medications,
prior imaging, past procedures, and prior auth history. This context is injected into the
LLM prompt so the clinical reasoning step can reference the patient's actual situation.

**Stream 2 — Policy documents (Azure AI Search)**
`policy_retrieval_node` constructs a hybrid search query from the CPT code, ICD-10 code,
and (if available) extracted document text. It retrieves the top-5 most relevant policy
chunks from a pre-indexed collection of 14 clinical policy PDFs. These chunks become
the authoritative grounding for the LLM — the system prompt instructs GPT-4o to cite
specific criteria from the retrieved chunks rather than relying on general knowledge.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | LangGraph 1.1 | 8-node stateful workflow with interrupt support |
| LLM | GPT-4o-mini (OpenAI) | Clinical reasoning, intent classification |
| Embeddings | text-embedding-3-small | Policy document vectorization |
| Vector search | Azure AI Search | Hybrid BM25 + HNSW policy retrieval |
| Database | Supabase (PostgreSQL) | Member records, case storage, audit trail |
| API | FastAPI + Uvicorn | REST endpoints for submit / status / review |
| PDF extraction | PyMuPDF (fitz) | Clinical attachment processing |
| Frontend | Vanilla HTML/CSS/JS | Nurse review portal (no framework) |
| Config | python-dotenv | Environment-based secret management |

---

## Key Design Decisions

**LangGraph interrupt for human-in-the-loop**
The graph is compiled with `interrupt_before=["human_review"]`. When the confidence
evaluation node decides a human is needed, LangGraph checkpoints the full state and
suspends before `human_review_node` runs. The workflow resumes only when the nurse
submits a decision via `POST /prior-auth/{case_id}/review`, which calls
`graph_app.invoke()` again with the same `thread_id`. No polling, no queues — just
LangGraph's native checkpoint mechanism.

**MemorySaver in development, AsyncPostgresSaver in production**
The checkpointer is the only infrastructure dependency for state persistence. Swapping
`MemorySaver` for `AsyncPostgresSaver` (backed by Supabase) requires a single argument
change in `build_graph()` — no graph topology changes.

**Confidence threshold routing**
`confidence_evaluation_node` applies four rules: (1) score < 0.75 → human review,
(2) decision is DENY → human review (nurse must confirm every denial), (3) decision is
INSUFFICIENT_INFO → human review, (4) `is_urgent=True` → human review regardless of
confidence. Only high-confidence APPROVEs auto-proceed.

**Enum serialization**
LangGraph's checkpoint serializer warned on Pydantic enum values. `clinical_reasoning_node`
converts the `AIDecision` enum to its `.value` string before returning — keeping the enum
for type safety inside the function while passing a plain string to the graph state.

**Single-file frontend**
The nurse portal is a single HTML file with embedded CSS and JavaScript. No build step,
no framework, no npm. Served by FastAPI's `FileResponse` at `/nurse-portal` with static
assets mounted at `/static`. This keeps the dev loop fast and the deployment trivial.

---

## Project Structure

```
prior-auth-system/
├── app/
│   ├── main.py                  # FastAPI app: submit, status, review endpoints
│   ├── graph/
│   │   ├── state.py             # PriorAuthState TypedDict (all 18 fields documented)
│   │   ├── nodes.py             # 8 LangGraph node functions
│   │   ├── routers.py           # 4 conditional edge routing functions
│   │   └── graph.py             # build_graph(), graph_app singleton, test_graph()
│   ├── models/
│   │   └── schemas.py           # Pydantic v2 request/response schemas
│   └── services/
│       ├── member_service.py    # Supabase member data (psycopg2, 8 sub-queries)
│       ├── rag_service.py       # Azure AI Search hybrid policy retrieval
│       └── doc_service.py       # PyMuPDF PDF extraction + clinical parsing
├── data/
│   ├── policies/                # 14 clinical policy PDFs (indexed in Azure AI Search)
│   ├── index_policies.py        # One-time script: vectorize + upload policy docs
│   └── load_data.py             # One-time script: load 57 synthetic members to Supabase
├── frontend/
│   └── index.html               # Nurse review portal (single-file, no framework)
├── output/                      # Synthea synthetic patient data (FHIR JSON)
├── requirements.txt
└── .env                         # API keys and connection strings (not committed)
```

---

## How It Works

### Submitting a request

```bash
curl -X POST http://localhost:8000/prior-auth/submit \
  -H "Content-Type: application/json" \
  -d '{
    "member_id": "06e30f00-63d9-7a7d-d30b-79f32641f372",
    "cpt_code": "72148",
    "icd_code": "M54.5",
    "provider_npi": "1234567890",
    "clinical_notes": "6 weeks low back pain with radiculopathy. Failed PT and NSAIDs.",
    "is_urgent": false,
    "case_type": "outpatient",
    "attachments": []
  }'
```

Response (auto-approved):
```json
{
  "case_id": "abc-123",
  "status": "APPROVED",
  "ai_recommendation": "APPROVE",
  "confidence_score": 0.91,
  "requires_human_review": false,
  "message": "Prior authorization approved based on clinical criteria.",
  "estimated_decision_time": "Immediate — auto-approved"
}
```

Response (needs nurse review):
```json
{
  "case_id": "def-456",
  "status": "PENDING_NURSE_REVIEW",
  "ai_recommendation": "DENY",
  "confidence_score": 0.62,
  "requires_human_review": true,
  "message": "Case flagged for nurse review.",
  "estimated_decision_time": "Within 72 hours"
}
```

### Nurse review

Navigate to `http://localhost:8000/nurse-portal` to open the review portal. Enter the
case ID, review the AI recommendation with criteria met/not met, and submit a decision
(Approve / Deny / Request More Info). If the nurse decision differs from the AI
recommendation, an override reason is required.

### Running locally

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY,
#          AZURE_SEARCH_INDEX, DATABASE_URL (Supabase connection string)

# 3. Start the API
uvicorn app.main:app --reload

# 4. Open the nurse portal
open http://localhost:8000/nurse-portal

# 5. (Optional) Smoke-test the full LangGraph pipeline
python -m app.graph.graph
```

---

## Data

**57 synthetic members** generated with [Synthea](https://github.com/synthetichealth/synthea)
and loaded into Supabase across 8 tables: members, insurance_plans, claims, conditions,
medications, imaging_studies, procedures, and prior_auth_requests. All member data is
entirely fictional.

**14 clinical policy documents** covering common prior auth scenarios (MRI, orthopedic
surgery, specialty medications, behavioral health, DME) indexed in Azure AI Search using
`text-embedding-3-small` vectors with HNSW approximate nearest-neighbor search.

---

## Status

This is a personal portfolio project demonstrating:
- Agentic AI workflow design with LangGraph
- Human-in-the-loop interrupt patterns
- RAG (retrieval-augmented generation) for grounded clinical reasoning
- Healthcare data modeling (FHIR-adjacent schemas, audit trails)
- Full-stack integration (FastAPI + Supabase + Azure + OpenAI)

Production readiness would require: HIPAA BAA agreements with all cloud vendors,
AsyncPostgresSaver replacing MemorySaver, proper auth/authz, rate limiting,
and a real-world policy document library reviewed by clinical staff.

---

## Disclaimer

This system is a **personal portfolio project** and is **not approved for any clinical use**.
All member data is synthetic. Policy documents are illustrative examples only and do not
represent any real insurer's coverage criteria. No real patient data is involved.
Prior authorization decisions in this system carry no medical or legal weight.
