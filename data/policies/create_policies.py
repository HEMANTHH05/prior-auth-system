# data/policies/create_policies.py
# Creates realistic policy documents for RAG indexing
# Based on real CMS coverage determination language

import os

# ─────────────────────────────────────────────────────
# Each policy document covers:
# - Which plan it applies to
# - Which procedure
# - Coverage criteria
# - Medical necessity requirements
# - Documentation required
# - Denial criteria
# ─────────────────────────────────────────────────────

policies = {

# ── MRI POLICIES ─────────────────────────────────────

"private_plans_mri_lumbar_spine.txt": """
POLICY DOCUMENT: MRI Lumbar Spine
APPLICABLE PLANS: Aetna, Anthem, Blue Cross Blue Shield, 
                  Cigna Health, Humana, UnitedHealthcare
PROCEDURE CODE: 72148 (MRI Lumbar Spine without contrast)
                72149 (MRI Lumbar Spine with contrast)
                72158 (MRI Lumbar Spine with and without contrast)
ICD-10 CODES: M54.5 (Low back pain), M51.1 (Lumbar disc degeneration),
              M47.816 (Spondylosis with radiculopathy lumbar region),
              M54.4 (Lumbago with sciatica)

COVERAGE CRITERIA:
MRI of the lumbar spine is covered when ALL of the following 
criteria are met:

1. CONSERVATIVE TREATMENT REQUIREMENT:
   - Patient must have completed minimum 6 weeks of conservative 
     treatment prior to imaging
   - Conservative treatment includes: physical therapy, chiropractic 
     care, rest, anti-inflammatory medications, or epidural steroid 
     injections
   - Documentation of failed conservative treatment must be present 
     in clinical notes
   - Exception: Conservative treatment requirement is waived for 
     patients with progressive neurological deficit, cauda equina 
     syndrome, or suspected malignancy

2. CLINICAL PRESENTATION REQUIREMENTS:
   - Documented low back pain with or without radiculopathy
   - Physical examination findings consistent with lumbar pathology
   - Neurological symptoms such as numbness, tingling, or weakness 
     in lower extremities

3. PRIOR IMAGING:
   - Plain radiograph (X-ray) of lumbar spine must be completed 
     prior to MRI unless contraindicated
   - X-ray results should be documented in clinical notes

4. MEDICAL NECESSITY DOCUMENTATION:
   - Treating physician must document medical necessity
   - Clinical notes must support imaging request
   - Symptoms must be of sufficient severity to warrant advanced imaging

DENIAL CRITERIA:
- Insufficient documentation of conservative treatment
- Plain radiograph not completed without documented contraindication
- No clinical documentation supporting medical necessity
- Routine screening without clinical indication
- Repeat imaging within 12 months without significant clinical change

DOCUMENTATION REQUIRED:
1. Physician clinical notes documenting symptoms and examination
2. Documentation of conservative treatment attempted and duration
3. Plain radiograph results or documented contraindication
4. Specific clinical question to be answered by imaging
5. Relevant medical history and current medications

AUTHORIZATION TIMEFRAME:
- Standard: Decision within 3 business days
- Urgent/Expedited: Decision within 24 hours
- Emergency: Retrospective review
""",

"medicare_mri_lumbar_spine.txt": """
POLICY DOCUMENT: MRI Lumbar Spine - Medicare Coverage
APPLICABLE PLANS: Medicare, Dual Eligible
PROCEDURE CODE: 72148, 72149, 72158
LCD NUMBER: L33762
ICD-10 CODES: M54.5, M51.1, M47.816, M54.4, M48.06

MEDICARE COVERAGE CRITERIA:
Medicare covers MRI of the lumbar spine when medically necessary
and the following criteria are met:

1. INDICATION REQUIREMENTS:
   - Low back pain with or without radiculopathy persisting 
     for 6 or more weeks despite conservative therapy
   - Suspected cord compression or cauda equina syndrome
   - Progressive neurological deficits
   - Suspected infection, tumor, or fracture
   - Pre-operative evaluation for spinal surgery

2. CONSERVATIVE TREATMENT:
   - Medicare requires documentation of at least 6 weeks of 
     conservative management
   - Acceptable conservative treatments: NSAIDs, acetaminophen, 
     muscle relaxants, physical therapy, chiropractic manipulation
   - Physician must document treatment response

3. PHYSICIAN DOCUMENTATION:
   - Order must come from treating physician
   - Clinical notes must be dated and signed
   - Notes must document examination findings
   - Functional limitations must be documented

4. TECHNICAL REQUIREMENTS:
   - Must be performed on FDA approved equipment
   - Radiologist interpretation required
   - Images must be of diagnostic quality

MEDICARE SPECIFIC REQUIREMENTS:
- Advanced Beneficiary Notice (ABN) required if coverage uncertain
- Claim must include appropriate diagnosis codes
- Medical record documentation must support medical necessity
- Physician signature required on order

NON-COVERED INDICATIONS:
- Screening without symptoms
- Repeat imaging without clinical change
- Patient convenience
- Research purposes without Medicare approval

APPEAL RIGHTS:
Medicare beneficiaries have the right to appeal coverage denials.
Redetermination requests must be filed within 120 days of denial.
""",

"medicaid_mri_lumbar_spine.txt": """
POLICY DOCUMENT: MRI Lumbar Spine - Medicaid Coverage
APPLICABLE PLANS: Medicaid
PROCEDURE CODE: 72148, 72149, 72158

MEDICAID COVERAGE CRITERIA:
Medicaid covers medically necessary MRI of the lumbar spine
when the following criteria are satisfied:

1. PRIOR AUTHORIZATION REQUIRED:
   - All MRI studies require prior authorization
   - Requests must be submitted before service delivery
   - Retroactive authorization not available except emergencies

2. MEDICAL NECESSITY CRITERIA:
   - Documented clinical indication for imaging
   - Conservative treatment trial of minimum 4 weeks
     (shorter than commercial plans due to Medicaid population)
   - Physician documentation of treatment failure
   - Clinical examination findings supporting request

3. PROVIDER REQUIREMENTS:
   - Ordering provider must be enrolled in Medicaid
   - Rendering facility must be Medicaid certified
   - Referral may be required based on managed care plan

4. DOCUMENTATION REQUIREMENTS:
   - Completed prior authorization request form
   - Clinical notes from past 90 days
   - Relevant imaging reports
   - Treatment history documentation

EXPEDITED REVIEW:
Available when standard timeline would seriously jeopardize
beneficiary health. Decision within 72 hours.

DENIAL AND APPEAL:
Denial notice provided within 2 business days.
Appeal rights explained in denial notice.
Fair hearing rights available.
""",

# ── PHYSICAL THERAPY POLICIES ─────────────────────────

"private_plans_physical_therapy.txt": """
POLICY DOCUMENT: Physical Therapy Services
APPLICABLE PLANS: Aetna, Anthem, Blue Cross Blue Shield,
                  Cigna Health, Humana, UnitedHealthcare
PROCEDURE CODES: 97110 (Therapeutic exercises)
                 97530 (Therapeutic activities)
                 97140 (Manual therapy)
                 97012 (Mechanical traction)
                 97035 (Ultrasound therapy)

COVERAGE CRITERIA:
Physical therapy is covered when ALL criteria are met:

1. MEDICAL NECESSITY:
   - Diagnosis requires skilled physical therapy intervention
   - Treatment goals are measurable and achievable
   - Patient has rehabilitation potential
   - Skilled therapist required (not maintenance therapy)

2. PHYSICIAN REFERRAL:
   - Valid prescription or referral from treating physician
   - Referral must include diagnosis and treatment goals
   - Some plans allow direct access for initial evaluation

3. TREATMENT PLAN:
   - Initial evaluation by licensed physical therapist
   - Documented plan of care with specific goals
   - Progress notes at minimum every 10 visits or 30 days
   - Re-authorization required for continued services

4. AUTHORIZED VISITS:
   - Initial authorization: 12 visits
   - Re-authorization available with documented progress
   - Maximum annual benefit varies by plan (typically 30-60 visits)
   - Visits must be medically necessary at each visit

DOCUMENTATION REQUIREMENTS:
1. Initial evaluation with objective measurements
2. Plan of care with specific measurable goals
3. Progress notes documenting response to treatment
4. Functional outcome measures
5. Discharge summary when care concludes

CONDITIONS COVERED:
- Post-surgical rehabilitation
- Musculoskeletal injuries and conditions
- Neurological rehabilitation
- Sports injuries
- Balance and fall prevention
- Back and neck pain
- Joint replacement rehabilitation

NOT COVERED:
- Maintenance therapy without skilled need
- Fitness or wellness programs
- Services provided by non-licensed personnel
- Duplicate services same day
""",

"medicare_physical_therapy.txt": """
POLICY DOCUMENT: Physical Therapy - Medicare Part B
APPLICABLE PLANS: Medicare, Dual Eligible
PROCEDURE CODES: 97110, 97530, 97140, 97012, 97035

MEDICARE PHYSICAL THERAPY COVERAGE:

1. COVERAGE REQUIREMENTS:
   - Services must be medically necessary
   - Must require skills of licensed therapist
   - Patient must have rehabilitation potential
   - Services must be reasonable in amount, frequency, duration

2. THERAPY CAPS AND EXCEPTIONS:
   - Medicare sets annual therapy cap amounts
   - Exception process available for medically necessary care
   - KX modifier required when cap exceeded
   - Documentation must support medical necessity for exceptions

3. SUPERVISION REQUIREMENTS:
   - Physical therapist must provide or supervise services
   - Assistants may provide services under PT supervision
   - Incident-to billing rules apply in some settings

4. DOCUMENTATION:
   - Plan of care signed by physician or non-physician practitioner
   - Therapy notes must document skilled need
   - Functional outcome reporting required
   - Progress reports every 10 treatment days

MEDICARE SPECIFIC BILLING:
- Claims submitted to Medicare Part B
- 20% coinsurance applies after deductible
- Secondary insurance may cover remaining cost
- Medigap policies vary in therapy coverage
""",

# ── SPECIALIST REFERRAL POLICIES ──────────────────────

"private_plans_specialist_referral.txt": """
POLICY DOCUMENT: Specialist Referral and Consultation
APPLICABLE PLANS: Aetna, Anthem, Blue Cross Blue Shield,
                  Cigna Health, Humana, UnitedHealthcare
PROCEDURE CODES: 99241-99245 (Office consultation)
                 99251-99255 (Inpatient consultation)

REFERRAL REQUIREMENTS:

1. PRIMARY CARE REFERRAL:
   - HMO plans: Referral from primary care physician required
   - PPO plans: No referral required for in-network specialists
   - POS plans: Referral required for HMO tier, optional for PPO tier

2. SPECIALIST QUALIFICATIONS:
   - Must be board certified or eligible in specialty
   - Must be credentialed with the health plan
   - In-network specialist preferred
   - Out-of-network requires prior authorization

3. PRIOR AUTHORIZATION:
   - Required for certain high-cost specialties:
     * Neurosurgery
     * Orthopedic surgery
     * Oncology
     * Transplant services
     * Bariatric surgery consultations

4. MEDICAL NECESSITY:
   - Primary care physician must document clinical reason
   - Specialist consultation must be medically indicated
   - Condition must be beyond scope of primary care

COVERED SERVICES:
- Initial consultation and evaluation
- Follow-up specialist visits
- Specialist-ordered diagnostic tests
- Specialist-recommended procedures (with separate auth)

AUTHORIZATION PROCESS:
- Submit referral request with clinical notes
- Typically decided within 2-3 business days
- Urgent requests within 24 hours
""",

# ── SURGICAL PROCEDURES POLICIES ─────────────────────

"private_plans_lumbar_surgery.txt": """
POLICY DOCUMENT: Lumbar Spine Surgery
APPLICABLE PLANS: Aetna, Anthem, Blue Cross Blue Shield,
                  Cigna Health, Humana, UnitedHealthcare
PROCEDURE CODES: 63030 (Laminotomy/discectomy)
                 63047 (Laminectomy)
                 22612 (Lumbar fusion)
                 22630 (Lumbar fusion posterior)

COVERAGE CRITERIA FOR LUMBAR SURGERY:

1. CONSERVATIVE TREATMENT FAILURE:
   - Minimum 6-12 weeks of conservative treatment
   - Must include physical therapy and medication management
   - Epidural steroid injections considered for radiculopathy
   - Documentation of treatment and outcome required

2. IMAGING REQUIREMENTS:
   - Recent MRI (within 12 months) confirming surgical pathology
   - MRI findings must correlate with clinical symptoms
   - X-rays required for fusion procedures
   - CT scan may be required for specific indications

3. CLINICAL CRITERIA:
   For Discectomy/Laminectomy:
   - Herniated disc or stenosis confirmed on imaging
   - Radiculopathy with neurological findings
   - Failed conservative treatment
   
   For Spinal Fusion:
   - Instability documented on imaging
   - Degenerative disc disease with failed conservative care
   - Spondylolisthesis with neurological symptoms
   - Revision surgery criteria

4. SECOND OPINION:
   - Required for all elective spinal fusion procedures
   - Second opinion physician must be independent
   - Second opinion must support surgical recommendation

5. PRE-OPERATIVE REQUIREMENTS:
   - Medical clearance from primary care physician
   - Anesthesia pre-operative evaluation
   - Smoking cessation counseling for fusion candidates
   - BMI optimization recommendations

DOCUMENTATION REQUIRED:
1. Complete surgical consultation notes
2. All imaging studies with radiology reports
3. Conservative treatment documentation
4. Second opinion report (for fusion)
5. Pre-operative history and physical

POST-OPERATIVE AUTHORIZATION:
- Physical therapy referral post-surgery
- Follow-up imaging may require separate authorization
- Pain management referral if needed
""",

"medicare_lumbar_surgery.txt": """
POLICY DOCUMENT: Lumbar Spine Surgery - Medicare Coverage
APPLICABLE PLANS: Medicare, Dual Eligible
PROCEDURE CODES: 63030, 63047, 22612, 22630

MEDICARE SURGICAL COVERAGE CRITERIA:

1. MEDICAL NECESSITY:
   - Surgery must be medically necessary
   - Non-surgical treatment must have been attempted
   - Clinical documentation must support surgical intervention
   - Benefits must outweigh risks

2. MEDICARE SPECIFIC REQUIREMENTS:
   - Pre-admission testing coverage
   - Surgeon must be Medicare enrolled
   - Facility must be Medicare certified
   - Anesthesia coverage requirements

3. INPATIENT VS OUTPATIENT:
   - Simple discectomy may be outpatient
   - Complex procedures typically inpatient
   - Two-midnight rule applies for inpatient stays
   - Observation status considerations

4. COVERAGE LIMITATIONS:
   - Experimental procedures not covered
   - Procedures without FDA approval
   - Services not reasonable and necessary

MEDICARE PAYMENT:
- DRG payment for inpatient procedures
- APC payment for outpatient procedures
- 20% coinsurance applies
- Additional costs for implants and devices
""",

# ── CARDIOLOGY POLICIES ───────────────────────────────

"private_plans_cardiac_stress_test.txt": """
POLICY DOCUMENT: Cardiac Stress Testing
APPLICABLE PLANS: Aetna, Anthem, Blue Cross Blue Shield,
                  Cigna Health, Humana, UnitedHealthcare
PROCEDURE CODES: 93015 (Treadmill stress test)
                 93351 (Stress echocardiogram)
                 78451 (Nuclear stress test SPECT)

COVERAGE CRITERIA:

1. INDICATIONS FOR STRESS TESTING:
   - Chest pain evaluation - typical or atypical angina
   - Shortness of breath with suspected cardiac etiology
   - Pre-operative cardiac risk assessment
   - Known coronary artery disease monitoring
   - Post-cardiac event risk stratification
   - Evaluation of exercise capacity

2. APPROPRIATE USE CRITERIA:
   - Intermediate pretest probability of coronary disease
   - New or changed symptoms in known CAD patient
   - Risk stratification after acute coronary syndrome
   - Evaluation of revascularization treatment

3. DOCUMENTATION REQUIRED:
   - Cardiology or internal medicine referral
   - Symptoms and duration documented
   - Risk factors documented (diabetes, hypertension, 
     hyperlipidemia, smoking, family history)
   - Resting ECG results
   - Recent laboratory results

4. MODALITY SELECTION:
   Standard Exercise Test: First-line for patients who can exercise
   Stress Echo: When resting ECG abnormal or better sensitivity needed
   Nuclear Stress: When other modalities inconclusive or high-risk

NOT COVERED:
- Routine screening in low-risk asymptomatic patients
- Repeat testing within 12 months without clinical change
- Pre-participation sports screening without symptoms
""",

# ── MENTAL HEALTH POLICIES ────────────────────────────

"all_plans_mental_health_services.txt": """
POLICY DOCUMENT: Mental Health and Behavioral Health Services
APPLICABLE PLANS: All Plans including Medicare and Medicaid
PROCEDURE CODES: 90837 (Individual psychotherapy 60 min)
                 90834 (Individual psychotherapy 45 min)
                 90847 (Family therapy with patient)
                 90853 (Group psychotherapy)
                 99213 (Office visit with psychiatric evaluation)

COVERAGE REQUIREMENTS:

1. MENTAL HEALTH PARITY:
   - Mental health benefits equal to medical/surgical benefits
   - Federal Mental Health Parity and Addiction Equity Act applies
   - No separate deductible or visit limits beyond medical benefits
   - Same prior authorization criteria as medical services

2. COVERED DIAGNOSES:
   - Major Depressive Disorder (F32, F33)
   - Anxiety Disorders (F40, F41)
   - Bipolar Disorder (F31)
   - Post-Traumatic Stress Disorder (F43.1)
   - Obsessive Compulsive Disorder (F42)
   - Schizophrenia and Psychotic Disorders (F20)
   - Substance Use Disorders (F10-F19)
   - ADHD (F90)
   - Eating Disorders (F50)

3. PROVIDER QUALIFICATIONS:
   - Licensed Psychologist (PhD, PsyD)
   - Licensed Clinical Social Worker (LCSW)
   - Licensed Professional Counselor (LPC)
   - Psychiatrist (MD, DO)
   - Advanced Practice Nurse (psychiatric specialty)

4. AUTHORIZATION REQUIREMENTS:
   - Initial evaluation typically covered without prior auth
   - Ongoing therapy may require authorization after initial visits
   - Inpatient psychiatric requires prior authorization
   - Partial hospitalization and intensive outpatient require auth

5. DOCUMENTATION:
   - DSM-5 diagnosis required
   - Treatment plan with measurable goals
   - Progress notes at each session
   - Medication management notes if applicable

CRISIS SERVICES:
- Emergency mental health services covered without prior auth
- Crisis stabilization services covered
- 988 Suicide and Crisis Lifeline available 24/7
""",

# ── DIABETES MANAGEMENT POLICIES ─────────────────────

"all_plans_diabetes_management.txt": """
POLICY DOCUMENT: Diabetes Management and Supplies
APPLICABLE PLANS: All Plans
PROCEDURE CODES: 95250 (Continuous glucose monitoring)
                 99213 (Diabetes management visit)
                 G0108 (Diabetes self-management training)

COVERAGE CRITERIA:

1. DIAGNOSIS REQUIREMENTS:
   - Type 1 Diabetes Mellitus (E10)
   - Type 2 Diabetes Mellitus (E11)
   - Gestational Diabetes (O24)
   - Pre-diabetes for prevention programs (R73.09)

2. DIABETES SUPPLIES COVERAGE:
   Blood Glucose Monitors:
   - One monitor per year covered
   - Test strips: 100/month for insulin users
   - Test strips: 50/month for non-insulin users
   
   Continuous Glucose Monitoring:
   - Covered for insulin-dependent diabetes
   - Must have history of hypoglycemia unawareness
   - Physician must order and document necessity
   
   Insulin Pumps:
   - Covered for Type 1 and insulin-dependent Type 2
   - Must demonstrate insulin-dependent status
   - Prior authorization required

3. DIABETES EDUCATION:
   - Initial diabetes self-management training covered
   - Annual follow-up education covered
   - Must be provided by accredited program
   - Physician referral required

4. PREVENTIVE SERVICES:
   - Annual diabetes screening for at-risk patients
   - Hemoglobin A1c testing covered quarterly if not at goal
   - Annual eye exam covered
   - Annual foot exam covered
   - Kidney function testing covered annually
"""
}


def create_policy_files():
    """
    Creates all policy document text files
    in data/policies/ folder.
    
    These files will be:
    1. Read by index_policies.py
    2. Chunked into smaller pieces
    3. Embedded as vectors
    4. Uploaded to Azure AI Search
    5. Queried by policy_retrieval_node
    """

    # Make sure directory exists
    os.makedirs('data/policies', exist_ok=True)

    print("Creating policy documents...")
    print("=" * 50)

    for filename, content in policies.items():
        filepath = f'data/policies/{filename}'
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ Created: {filename}")

    print("=" * 50)
    print(f"✅ {len(policies)} policy documents created")
    print("📁 Location: data/policies/")


if __name__ == '__main__':
    create_policy_files()