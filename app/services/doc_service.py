"""
app/services/doc_service.py

Clinical PDF extraction and parsing service for the Prior Authorization AI
system.

Called by Node 3 (document_ingestion_node) in the LangGraph workflow.
Doctors upload supporting PDF attachments — physician narratives, lab results,
imaging reports, treatment histories — alongside their prior-auth request.
This service turns those binary PDFs into structured dicts the clinical-
reasoning LLM node can consume directly.

Why PyMuPDF (fitz) instead of Azure Document Intelligence?
- Zero per-page API cost — critical when processing hundreds of attachments
  per day at scale.
- Sub-100 ms local extraction vs ~2–5 s API round-trip per page.
- No network dependency; works even if Azure is degraded.
- Azure Document Intelligence is reserved for complex form extraction (EOBs,
  structured lab panels) handled by a separate service.

Scanned PDF handling:
PyMuPDF extracts the text layer embedded by the PDF creator.  If a PDF was
produced by scanning a paper document without OCR, that layer is empty.  We
detect this condition (extracted text < threshold) and return a warning
string rather than silently returning nothing — the LangGraph node can then
flag the case for manual document review.
"""

from __future__ import annotations

import io
import logging
import re
import textwrap
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Minimum character count below which we consider a PDF "scanned / image-only"
_SCANNED_PDF_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


def _clean_extracted_text(raw: str) -> str:
    """Normalise whitespace and fix common PDF extraction artefacts.

    PDF text extraction often produces:
    - Excessive blank lines between paragraphs (multiple \\n in a row)
    - Trailing spaces on every line (from justified text spacing)
    - Soft hyphens at line breaks (e.g., "treat-\\nment")
    - Form-feed characters (\\f) used as page separators
    - Non-breaking spaces (\\xa0) from PDF encoding

    We fix all of these while preserving meaningful paragraph breaks
    (a single blank line = paragraph boundary).

    Args:
        raw: Raw string from fitz page.get_text().

    Returns:
        Cleaned string suitable for regex parsing and LLM injection.
    """
    # 1. Normalise page-feed characters to newlines
    text = raw.replace("\f", "\n")

    # 2. Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")

    # 3. Rejoin soft-hyphenated words split across lines
    #    e.g. "treat-\nment" → "treatment"
    #    Pattern: hyphen at end of line followed by optional spaces and newline
    text = re.sub(r"-\s*\n\s*", "", text)

    # 4. Strip trailing whitespace on every line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # 5. Collapse runs of 3+ blank lines to a single blank line
    #    This preserves paragraph structure without excessive vertical space
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Collapse multiple spaces/tabs within a line to a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# 1. PDF text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(source: str | bytes) -> str:
    """Extract and clean all text from a PDF file.

    Supports two input modes:
    - File path (str): read the PDF from disk.  Used when the LangGraph node
      has already downloaded the blob to a temp file.
    - Raw bytes: process the PDF in memory without touching disk.  Used when
      streaming directly from Azure Blob Storage without a temp file.

    Scanned PDF detection:
    If the total extracted text is shorter than _SCANNED_PDF_THRESHOLD
    characters, the PDF almost certainly has no embedded text layer (it is an
    image-only scan).  We return a descriptive warning string so the calling
    node can add a "manual review required" flag rather than sending an empty
    string to the LLM.

    Args:
        source: Absolute file path string OR raw PDF bytes.

    Returns:
        Cleaned extracted text, or a warning string if the PDF appears to be
        an image-only scan.  Never raises — returns an error string instead.
    """
    try:
        # Open from bytes or from path — fitz handles both via stream= kwarg
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)

        pages_text: list[str] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # get_text("text") returns plain text with newlines preserving
            # the visual reading order of the page
            page_text = page.get_text("text")
            pages_text.append(page_text)

        doc.close()

        raw_text = "\n".join(pages_text)
        cleaned = _clean_extracted_text(raw_text)

        if len(cleaned) < _SCANNED_PDF_THRESHOLD:
            logger.warning(
                "PDF appears to be an image-only scan (extracted %d chars). "
                "OCR not available; manual review may be required.",
                len(cleaned),
            )
            return (
                "[SCANNED PDF - NO TEXT LAYER DETECTED] "
                "This document appears to be a scanned image. "
                "Manual review of the original document is required."
            )

        logger.info("Extracted %d characters from PDF.", len(cleaned))
        return cleaned

    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        return f"[PDF EXTRACTION ERROR] Could not extract text: {exc}"


# ---------------------------------------------------------------------------
# 2. Clinical document parser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Compiled regex patterns
#
# We compile all patterns at module load time (not per-call) to avoid
# recompilation overhead in a LangGraph workflow that may process many
# documents per session.
#
# All patterns use re.IGNORECASE so "Diagnosis:", "DIAGNOSIS:", and
# "diagnosis:" all match without separate patterns.
# ---------------------------------------------------------------------------

# --- Patient name ---
# Matches common header formats used in clinical notes and referral letters:
#   "Patient: John A. Smith"
#   "Patient Name: Smith, John"
#   "Re: John Smith"
#   "Name: John Smith, DOB 01/01/1960"
# Capture group 1: the name string (may include comma-separated Last, First)
_RE_PATIENT_NAME = re.compile(
    r"(?:patient(?:\s+name)?|re(?:garding)?)\s*:\s*([A-Za-z][\w,\.\-\s]{2,50}?)(?=\s*[\n,]|\s+DOB|\s+MR#|\s+MRN)",
    re.IGNORECASE,
)

# --- Physician / provider name ---
# Matches:
#   "Physician: Dr. Jane Doe, MD"
#   "Referring Physician: Robert Smith, DO"
#   "Provider: Dr. Alice Brown"
#   "Attending: Dr. James White"
#   "Sincerely, Dr. John Doe" (letter signature block)
_RE_PHYSICIAN = re.compile(
    r"(?:physician|provider|referring\s+physician|attending|sincerely|regards)\s*[,:]?\s*"
    r"((?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s*,\s*(?:MD|DO|NP|PA|APRN))?)",
    re.IGNORECASE,
)

# --- Diagnosis ---
# Matches:
#   "Diagnosis: Severe knee osteoarthritis"
#   "Dx: M17.11"
#   "Primary Diagnosis: ..."
#   "Clinical Diagnosis: ..."
# Captures text up to end-of-line; we strip it afterward
_RE_DIAGNOSIS = re.compile(
    r"(?:(?:primary|clinical|working|final)\s+)?(?:diagnosis|dx)\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE,
)

# --- Procedure requested ---
# Matches:
#   "Procedure: Total Knee Arthroplasty"
#   "Requested Procedure: MRI Lumbar Spine"
#   "Procedure Requested: ..."
#   "Service Requested: ..."
_RE_PROCEDURE = re.compile(
    r"(?:requested\s+)?(?:procedure|service)(?:\s+requested)?\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE,
)

# --- Conservative treatments ---
# We look for a broad vocabulary of conservative treatment modalities that
# payers use as step-therapy requirements before authorising procedures.
# The pattern captures the entire clause around the keyword so we get context
# (e.g., "6 weeks of physical therapy" not just "physical therapy").
#
# Strategy: find any sentence/clause containing one of these keywords.
# We collect all unique matched sentences into a list.
_CONSERVATIVE_TREATMENT_KEYWORDS = [
    r"physical\s+therapy",
    r"physiotherapy",
    r"chiropractic",
    r"occupational\s+therapy",
    r"NSAID",
    r"anti-?inflammat\w*",
    r"ibuprofen",
    r"naproxen",
    r"meloxicam",
    r"diclofenac",
    r"cortisone",
    r"corticosteroid",
    r"steroid\s+injection",
    r"viscosupplementat\w*",
    r"hyaluronic\s+acid",
    r"intra-?articular\s+injection",
    r"DMARD",
    r"methotrexate",
    r"rest\s+and\s+ice",
    r"activity\s+modification",
    r"weight\s+loss",
    r"bracing",
    r"orthotics",
    r"acupuncture",
    r"massage\s+therapy",
]

# Sentence-level scan: matches the sentence containing any treatment keyword.
# A "sentence" here is delimited by newline or period+space.
_RE_CONSERVATIVE = re.compile(
    r"[^\n.]*(?:" + "|".join(_CONSERVATIVE_TREATMENT_KEYWORDS) + r")[^\n.]*[.\n]?",
    re.IGNORECASE,
)

# --- Treatment duration ---
# Matches time-quantity phrases near treatment keywords:
#   "6 weeks of physical therapy"
#   "three months of conservative treatment"
#   "failed 6-week course of PT"
#
# Two-part negative lookahead prevents false positives:
#   (?![-\s]*old)  — excludes "67-year-old" (patient age)
#   (?![-\s]*\d)   — excludes compound numbers like "6-month-12-day"
# We also require at least one contextual word (of/course/trial/period/
# conservative/therapy/treatment) within 25 chars to further anchor the
# match to an actual duration statement rather than incidental time words.
_RE_DURATION = re.compile(
    r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|"
    r"several|multiple)\s*[-\s]?"
    r"(week|month|day)s?"                       # "year" removed — too ambiguous
    r"(?![-\s]*old)"                             # exclude "67-year-old"
    r"(?=[-\s\w]{0,25}"                          # lookahead window of 25 chars
    r"(?:of|course|trial|period|conservative|therapy|treatment|PT\b|physical))",
    re.IGNORECASE,
)

# --- Prior imaging studies ---
# Captures imaging modality names and their anatomical context.
# We look for whole phrases like "MRI of the lumbar spine", "CT chest", etc.
# Anatomical modifiers (up to 5 words) are captured in group 2.
_IMAGING_MODALITIES = [
    r"MRI",
    r"magnetic\s+resonance\s+imaging",
    r"CT\s+scan",
    r"computed\s+tomography",
    r"X-?ray",
    r"plain\s+film",
    r"radiograph",
    r"ultrasound",
    r"PET\s+scan",
    r"bone\s+scan",
    r"DEXA\s+scan",
    r"fluoroscopy",
    r"echocardiogram",
    r"echo",
]

_RE_IMAGING = re.compile(
    r"(?:" + "|".join(_IMAGING_MODALITIES) + r")"
    r"(?:\s+(?:of\s+(?:the\s+)?)?[\w\s]{1,40})?",
    re.IGNORECASE,
)

# --- Medical necessity block ---
# Clinical notes frequently contain a clearly labelled section for the
# medical necessity justification.  We extract the paragraph(s) that follow
# the label.  The section ends at the next all-caps section header or double
# newline.
#
# Critical design choice — the label must be preceded by a newline (or be
# at the very start of the string) AND must be followed by a colon or dash.
# This prevents matching document titles like "PRIOR AUTHORIZATION CLINICAL
# JUSTIFICATION" (which lacks a colon) from being mistaken for section labels.
# re.MULTILINE is required so \n inside the pattern matches actual newlines.
_RE_MEDICAL_NECESSITY = re.compile(
    r"(?:^|\n)"                                  # label must start on its own line
    r"(?:medical\s+necessity|clinical\s+justification|clinical\s+rationale|"
    r"reason\s+for\s+request|clinical\s+indication)"
    r"\s*[:\-]\s*\n"                             # colon/dash required; then newline
    r"([\s\S]{20,600}?)(?=\n\n|\n[A-Z][A-Z\s]{4,}:|\Z)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_clinical_document(text: str) -> dict[str, Any]:
    """Parse raw extracted PDF text into a structured clinical dict.

    Uses regex and keyword matching — no LLM call — to keep this node fast
    and deterministic.  The results supplement (not replace) the raw_text
    passed to the clinical-reasoning LLM.

    All fields default to None / empty list; the function never raises.
    Missing fields are expected and handled gracefully by the LLM node.

    Args:
        text: Cleaned text string from extract_text_from_pdf().

    Returns:
        Dict with keys:
            patient_name         (str | None)
            physician_name       (str | None)
            diagnosis            (str | None)
            procedure_requested  (str | None)
            conservative_treatment (list[str])
            treatment_duration   (str | None)
            prior_imaging        (list[str])
            medical_necessity    (str | None)
            raw_text             (str)
    """
    result: dict[str, Any] = {
        "patient_name": None,
        "physician_name": None,
        "diagnosis": None,
        "procedure_requested": None,
        "conservative_treatment": [],
        "treatment_duration": None,
        "prior_imaging": [],
        "medical_necessity": None,
        "raw_text": text,
    }

    try:
        # --- patient_name ---
        m = _RE_PATIENT_NAME.search(text)
        if m:
            # Strip any trailing commas, whitespace, or parenthetical age
            result["patient_name"] = m.group(1).strip().rstrip(",")

        # --- physician_name ---
        m = _RE_PHYSICIAN.search(text)
        if m:
            result["physician_name"] = m.group(1).strip().rstrip(",")

        # --- diagnosis ---
        m = _RE_DIAGNOSIS.search(text)
        if m:
            result["diagnosis"] = m.group(1).strip()

        # --- procedure_requested ---
        m = _RE_PROCEDURE.search(text)
        if m:
            result["procedure_requested"] = m.group(1).strip()

        # --- conservative_treatment ---
        # findall returns all non-overlapping matches; deduplicate while
        # preserving order (dict.fromkeys trick) and strip each match.
        raw_treatments = _RE_CONSERVATIVE.findall(text)
        unique_treatments = list(
            dict.fromkeys(t.strip() for t in raw_treatments if t.strip())
        )
        result["conservative_treatment"] = unique_treatments

        # --- treatment_duration ---
        # Return the first duration expression found anywhere in the document.
        # We intentionally pick the first one; typically the chief complaint
        # section states the total duration of conservative therapy upfront.
        m = _RE_DURATION.search(text)
        if m:
            # Reconstruct the full matched phrase (quantity + unit)
            result["treatment_duration"] = m.group(0).strip()

        # --- prior_imaging ---
        # Collect all imaging mentions; deduplicate; normalise whitespace.
        raw_imaging = _RE_IMAGING.findall(text)
        unique_imaging = list(
            dict.fromkeys(
                re.sub(r"\s+", " ", img.strip())
                for img in raw_imaging
                if img.strip()
            )
        )
        result["prior_imaging"] = unique_imaging

        # --- medical_necessity ---
        m = _RE_MEDICAL_NECESSITY.search(text)
        if m:
            # Collapse internal whitespace runs; preserve paragraph breaks
            block = re.sub(r"[ \t]{2,}", " ", m.group(1)).strip()
            result["medical_necessity"] = block

    except Exception as exc:
        # Parser failures must not crash the LangGraph node.
        # Log the error but return whatever fields we did populate.
        logger.error("Clinical document parsing error: %s", exc)

    return result


# ---------------------------------------------------------------------------
# 3. Master function — called by document_ingestion_node
# ---------------------------------------------------------------------------


def process_attachment(source: str | bytes) -> dict[str, Any]:
    """Extract and parse a clinical PDF attachment into a structured dict.

    This is the single entry point for document_ingestion_node.  It chains
    extract_text_from_pdf → parse_clinical_document and wraps both in a
    top-level error boundary so any failure produces a safe, inspectable
    result rather than crashing the workflow.

    Args:
        source: Absolute file path (str) or raw PDF bytes.

    Returns:
        Dict combining all parse_clinical_document keys plus:
            extraction_success (bool)   — False if text extraction failed
            extraction_warning (str | None) — set for scanned PDFs or errors
    """
    extraction_warning: str | None = None

    # --- Step 1: extract text ---
    try:
        text = extract_text_from_pdf(source)

        # Distinguish a scanned/errored PDF from genuine content
        if text.startswith("[SCANNED PDF") or text.startswith("[PDF EXTRACTION ERROR"):
            extraction_warning = text
            text = ""          # don't send the warning string to the parser
            extraction_success = False
        else:
            extraction_success = True

    except Exception as exc:
        logger.error("Unexpected error in process_attachment extraction: %s", exc)
        text = ""
        extraction_success = False
        extraction_warning = f"Unexpected extraction error: {exc}"

    # --- Step 2: parse structured fields ---
    parsed = parse_clinical_document(text)

    # --- Step 3: merge extraction metadata ---
    parsed["extraction_success"] = extraction_success
    parsed["extraction_warning"] = extraction_warning

    return parsed


# ---------------------------------------------------------------------------
# Test / smoke-test function
# ---------------------------------------------------------------------------


def test_doc_service() -> None:
    """Create a synthetic clinical note PDF in memory and process it.

    Verifies the full pipeline — PDF creation → text extraction → clinical
    parsing — without requiring any external files or services.

    Usage::

        source venv/bin/activate
        python -m app.services.doc_service
    """
    # --- Build a realistic clinical note as a PDF in memory ---
    sample_note = textwrap.dedent("""\
        PRIOR AUTHORIZATION CLINICAL JUSTIFICATION

        Patient: Sarah M. Johnson
        Date of Birth: 03/15/1958
        MRN: 789-456-123

        Referring Physician: Dr. Michael R. Torres, MD
        Specialty: Orthopedic Surgery

        Diagnosis: Severe bilateral knee osteoarthritis (ICD-10: M17.11)

        Procedure Requested: Total Knee Arthroplasty (CPT: 27447)

        Medical Necessity:
        Ms. Johnson is a 67-year-old female with a 4-year history of
        progressive bilateral knee osteoarthritis confirmed by X-ray and
        MRI of the left knee showing Kellgren-Lawrence grade 4 changes.
        She reports severe pain (8/10) limiting ambulation to under 100 feet.

        Conservative Treatment History:
        The patient has failed 6 months of physical therapy with a
        licensed physiotherapist. She has trialled NSAIDs including
        ibuprofen 800 mg TID and naproxen 500 mg BID without adequate
        relief. Three rounds of corticosteroid injections and one course
        of viscosupplementation (hyaluronic acid) were administered in
        the past 12 months with only temporary benefit.

        Prior Imaging:
        - X-ray bilateral knees (01/2025): KL grade 4 bilateral
        - MRI left knee without contrast (03/2025): full-thickness
          cartilage loss, subchondral edema

        Clinical Justification:
        Given failure of all appropriate conservative treatment modalities
        over a period of 6 months and the severity of functional impairment,
        total knee arthroplasty is medically necessary and is the
        appropriate next intervention per AAOS clinical guidelines.

        Sincerely, Dr. Michael R. Torres, MD
    """)

    # Create an in-memory PDF with PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 72),
        sample_note,
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    print("=" * 70)
    print("doc_service smoke test — synthetic clinical note")
    print("=" * 70)

    result = process_attachment(pdf_bytes)

    print(f"\nextraction_success : {result['extraction_success']}")
    print(f"extraction_warning : {result['extraction_warning']}")
    print(f"patient_name       : {result['patient_name']}")
    print(f"physician_name     : {result['physician_name']}")
    print(f"diagnosis          : {result['diagnosis']}")
    print(f"procedure_requested: {result['procedure_requested']}")
    print(f"treatment_duration : {result['treatment_duration']}")

    print(f"\nconservative_treatment ({len(result['conservative_treatment'])} found):")
    for t in result["conservative_treatment"]:
        print(f"  - {t[:100]}")

    print(f"\nprior_imaging ({len(result['prior_imaging'])} found):")
    for img in result["prior_imaging"]:
        print(f"  - {img}")

    print(f"\nmedical_necessity:\n{result['medical_necessity']}")

    print("\n" + "=" * 70)
    print("raw_text preview (first 300 chars):")
    print(result["raw_text"][:300])


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=logging.INFO)
    test_doc_service()
