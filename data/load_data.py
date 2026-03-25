# data/load_data.py
# ─────────────────────────────────────────────────────
# PURPOSE: Load Synthea CSV data into Supabase PostgreSQL
# RUN ONCE: python3 data/load_data.py
# ─────────────────────────────────────────────────────

# ── IMPORTS ──────────────────────────────────────────
import pandas as pd        # reads CSV files into dataframes
import psycopg2            # connects to PostgreSQL
import os                  # reads environment variables
import uuid                # generates unique IDs
import numpy as np         # handles NaN values from CSV
from dotenv import load_dotenv  # loads .env file
from datetime import datetime   # handles date conversions

# Load .env file so we can read DATABASE_URL
# Without this os.getenv returns None
load_dotenv()

# ── DATABASE CONNECTION ───────────────────────────────
def get_connection():
    """
    Creates and returns a PostgreSQL connection.
    
    Uses DATABASE_URL from .env file.
    Called before each loading function.
    Closed after each loading function.
    
    Why not one global connection?
    → If one table fails, others still work
    → Clean error handling per table
    """
    return psycopg2.connect(os.getenv('DATABASE_URL'))


# ── HELPER FUNCTIONS ──────────────────────────────────
def clean_value(value):
    """
    Converts pandas NaN/None to Python None.
    
    Why needed?
    → CSV empty cells become pandas NaN
    → PostgreSQL doesn't understand NaN
    → Must convert to None (SQL NULL)
    
    Example:
    NaN → None
    "hello" → "hello"
    "" → None
    123 → 123
    """
    if pd.isna(value) or value == '':
        return None
    return value


def clean_uuid(value):
    """
    Validates and cleans UUID values.
    
    Why needed?
    → Synthea UUIDs are valid
    → But empty cells must become None
    → PostgreSQL UUID columns reject empty strings
    
    Example:
    "06e30f00-63d9-7a7d-d30b-79f32641f372" → same
    "" → None
    NaN → None
    """
    if pd.isna(value) or value == '':
        return None
    return str(value)


def clean_decimal(value):
    """
    Converts string numbers to float.
    
    Why needed?
    → CSV reads numbers as strings
    → PostgreSQL DECIMAL needs float
    → Empty cells need to be None
    
    Example:
    "9082.66" → 9082.66
    "" → None
    NaN → None
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except:
        return None


def clean_integer(value):
    """
    Converts string integers to int.
    Same reason as clean_decimal.
    
    Example:
    "40913" → 40913
    "" → None
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return int(float(value))
    except:
        return None


def clean_datetime(value):
    """
    Converts Synthea datetime strings to Python datetime.
    
    Why needed?
    → Synthea format: "2022-05-17T21:37:11Z"
    → PostgreSQL needs proper datetime object
    → Empty cells need to be None
    
    Example:
    "2022-05-17T21:37:11Z" → datetime(2022, 5, 17, 21, 37, 11)
    "" → None
    """
    if pd.isna(value) or value == '':
        return None
    try:
        # Remove Z at end, parse ISO format
        return datetime.fromisoformat(
            str(value).replace('Z', '+00:00')
        )
    except:
        return None


def clean_date(value):
    """
    Converts date strings to Python date.
    
    Synthea date format: "2022-05-17"
    
    Example:
    "2022-05-17" → date(2022, 5, 17)
    "" → None
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return datetime.strptime(
            str(value)[:10], '%Y-%m-%d'
        ).date()
    except:
        return None


# ── LOADING FUNCTIONS ─────────────────────────────────
# One function per table
# Each function:
# 1. Reads its CSV file
# 2. Cleans each column
# 3. Inserts into PostgreSQL
# 4. Prints progress
# ─────────────────────────────────────────────────────

def load_insurance_plans():
    """
    Loads payers.csv → insurance_plans table
    
    Must load FIRST because:
    → members_insurance references this table
    → encounters references this table
    → claims references this table
    """
    print("\n Loading insurance_plans...")

    # Read CSV file
    # dtype=str → read everything as string first
    # we clean types ourselves
    df = pd.read_csv(
        'output/csv/payers.csv',
        dtype=str
    )

    print(f"   Found {len(df)} insurance plans")

    # Connect to database
    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        # _ means we don't need the row index
        # row contains all column values
        try:
            cur.execute("""
                INSERT INTO insurance_plans (
                    id, name, ownership,
                    address, city, state, zip, phone,
                    amount_covered, amount_uncovered,
                    revenue, unique_customers
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                # %s placeholders replaced by these values
                # in exact same order as column list above
                clean_uuid(row['Id']),
                clean_value(row['NAME']),
                clean_value(row['OWNERSHIP']),
                clean_value(row['ADDRESS']),
                clean_value(row['CITY']),
                clean_value(row['STATE_HEADQUARTERED']),
                clean_value(row['ZIP']),
                clean_value(row['PHONE']),
                clean_decimal(row['AMOUNT_COVERED']),
                clean_decimal(row['AMOUNT_UNCOVERED']),
                clean_decimal(row['REVENUE']),
                clean_integer(row['UNIQUE_CUSTOMERS'])
            ))
            success += 1
        except Exception as e:
            failed += 1
            print(f"   Row failed: {e}")

    # Commit saves all inserts permanently
    # Without commit → nothing saved
    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_members():
    """
    Loads patients.csv → members table
    
    Must load SECOND because:
    → almost every other table references members
    """
    print("\n Loading members...")

    df = pd.read_csv(
        'output/csv/patients.csv',
        dtype=str
    )

    print(f"   Found {len(df)} members")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO members (
                    id, birthdate, deathdate,
                    ssn, prefix, first_name,
                    middle_name, last_name, suffix,
                    marital_status, race, ethnicity,
                    gender, birthplace, address,
                    city, state, county, zip,
                    lat, lon,
                    healthcare_expenses,
                    healthcare_coverage, income
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                clean_uuid(row['Id']),
                clean_date(row['BIRTHDATE']),
                clean_date(row['DEATHDATE']),
                clean_value(row['SSN']),
                clean_value(row['PREFIX']),
                clean_value(row['FIRST']),
                clean_value(row['MIDDLE']),
                clean_value(row['LAST']),
                clean_value(row['SUFFIX']),
                clean_value(row['MARITAL']),
                clean_value(row['RACE']),
                clean_value(row['ETHNICITY']),
                clean_value(row['GENDER']),
                clean_value(row['BIRTHPLACE']),
                clean_value(row['ADDRESS']),
                clean_value(row['CITY']),
                clean_value(row['STATE']),
                clean_value(row['COUNTY']),
                clean_value(row['ZIP']),
                clean_decimal(row['LAT']),
                clean_decimal(row['LON']),
                clean_decimal(row['HEALTHCARE_EXPENSES']),
                clean_decimal(row['HEALTHCARE_COVERAGE']),
                clean_integer(row['INCOME'])
            ))
            success += 1
        except Exception as e:
            failed += 1
            print(f"   Row failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_providers():
    """
    Loads providers.csv → providers table

    Must load THIRD because:
    → encounters references providers
    → claims references providers
    """
    print("\n Loading providers...")

    df = pd.read_csv(
        'output/csv/providers.csv',
        dtype=str
    )

    print(f"   Found {len(df)} providers")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO providers (
                    id, organization_id, name,
                    gender, speciality, address,
                    city, state, zip,
                    lat, lon,
                    total_encounters, total_procedures
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                clean_uuid(row['Id']),
                clean_uuid(row['ORGANIZATION']),
                clean_value(row['NAME']),
                clean_value(row['GENDER']),
                clean_value(row['SPECIALITY']),
                clean_value(row['ADDRESS']),
                clean_value(row['CITY']),
                clean_value(row['STATE']),
                clean_value(row['ZIP']),
                clean_decimal(row['LAT']),
                clean_decimal(row['LON']),
                clean_integer(row['ENCOUNTERS']),
                clean_integer(row['PROCEDURES'])
            ))
            success += 1
        except Exception as e:
            failed += 1
            print(f"   Row failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_member_insurance():
    """
    Loads payer_transitions.csv → member_insurance table

    Calculates is_active:
    → end_date in past = inactive
    → end_date in future or NULL = active
    """
    print("\n Loading member_insurance...")

    df = pd.read_csv(
        'output/csv/payer_transitions.csv',
        dtype=str
    )

    print(f"   Found {len(df)} insurance records")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0
    now = datetime.now()

    for _, row in df.iterrows():
        try:
            # Calculate is_active ourselves
            # since we removed generated column
            end_date = clean_datetime(row['END_DATE'])
            is_active = end_date is None or end_date.replace(
                tzinfo=None
            ) > now

            cur.execute("""
                INSERT INTO member_insurance (
                    member_id, member_policy_id,
                    plan_id, start_date, end_date,
                    plan_ownership, owner_name,
                    is_active
                ) VALUES (
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s
                )
            """, (
                clean_uuid(row['PATIENT']),
                clean_uuid(row['MEMBERID']),
                clean_uuid(row['PAYER']),
                clean_datetime(row['START_DATE']),
                end_date,
                clean_value(row['PLAN_OWNERSHIP']),
                clean_value(row['OWNER_NAME']),
                is_active
            ))
            success += 1
        except Exception as e:
            failed += 1
            print(f"   Row failed: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_encounters():
    """
    Loads encounters.csv → encounters table

    Every doctor visit, hospital stay,
    wellness check is an encounter.
    Many other tables link back to encounters.
    """
    print("\n Loading encounters...")

    df = pd.read_csv(
        'output/csv/encounters.csv',
        dtype=str
    )

    print(f"   Found {len(df)} encounters")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO encounters (
                    id, start_time, stop_time,
                    member_id, provider_id, plan_id,
                    encounter_class, code, description,
                    base_cost, total_claim_cost,
                    payer_coverage,
                    reason_code, reason_description
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                clean_uuid(row['Id']),
                clean_datetime(row['START']),
                clean_datetime(row['STOP']),
                clean_uuid(row['PATIENT']),
                clean_uuid(row['PROVIDER']),
                clean_uuid(row['PAYER']),
                clean_value(row['ENCOUNTERCLASS']),
                clean_value(row['CODE']),
                clean_value(row['DESCRIPTION']),
                clean_decimal(row['BASE_ENCOUNTER_COST']),
                clean_decimal(row['TOTAL_CLAIM_COST']),
                clean_decimal(row['PAYER_COVERAGE']),
                clean_value(row['REASONCODE']),
                clean_value(row['REASONDESCRIPTION'])
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_claims():
    """
    Loads claims.csv → claims table

    Insurance claims filed for each encounter.
    Contains diagnosis codes (ICD equivalent).
    """
    print("\n Loading claims...")

    df = pd.read_csv(
        'output/csv/claims.csv',
        dtype=str
    )

    print(f"   Found {len(df)} claims")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO claims (
                    id, member_id, provider_id,
                    primary_insurance_id,
                    secondary_insurance_id,
                    diagnosis1, diagnosis2, diagnosis3,
                    diagnosis4, diagnosis5, diagnosis6,
                    diagnosis7, diagnosis8,
                    service_date,
                    status1, status2,
                    outstanding1, outstanding2,
                    last_billed_date
                ) VALUES (
                    %s, %s, %s,
                    %s,
                    %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s,
                    %s, %s,
                    %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                clean_uuid(row['Id']),
                clean_uuid(row['PATIENTID']),
                clean_uuid(row['PROVIDERID']),
                clean_uuid(row['PRIMARYPATIENTINSURANCEID']),
                clean_uuid(row['SECONDARYPATIENTINSURANCEID']),
                clean_value(row['DIAGNOSIS1']),
                clean_value(row['DIAGNOSIS2']),
                clean_value(row['DIAGNOSIS3']),
                clean_value(row['DIAGNOSIS4']),
                clean_value(row['DIAGNOSIS5']),
                clean_value(row['DIAGNOSIS6']),
                clean_value(row['DIAGNOSIS7']),
                clean_value(row['DIAGNOSIS8']),
                clean_datetime(row['SERVICEDATE']),
                clean_value(row['STATUS1']),
                clean_value(row['STATUS2']),
                clean_decimal(row['OUTSTANDING1']),
                clean_decimal(row['OUTSTANDING2']),
                clean_datetime(row['LASTBILLEDDATE1'])
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_procedures():
    """
    Loads procedures.csv → procedures table

    Every medical procedure performed.
    Critical for prior auth:
    "has this patient had this procedure before?"
    """
    print("\n Loading procedures...")

    df = pd.read_csv(
        'output/csv/procedures.csv',
        dtype=str
    )

    print(f"   Found {len(df)} procedures")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO procedures (
                    start_time, stop_time,
                    member_id, encounter_id,
                    system, code, description,
                    base_cost,
                    reason_code, reason_description
                ) VALUES (
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s,
                    %s, %s
                )
            """, (
                clean_datetime(row['START']),
                clean_datetime(row['STOP']),
                clean_uuid(row['PATIENT']),
                clean_uuid(row['ENCOUNTER']),
                clean_value(row['SYSTEM']),
                clean_value(row['CODE']),
                clean_value(row['DESCRIPTION']),
                clean_decimal(row['BASE_COST']),
                clean_value(row['REASONCODE']),
                clean_value(row['REASONDESCRIPTION'])
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_conditions():
    """
    Loads conditions.csv → conditions table

    Patient diagnoses — active and resolved.
    Critical for:
    → RAG query construction
    → Clinical reasoning context
    → "What condition justifies this procedure?"

    is_active calculated:
    → stop_date is NULL = still active
    → stop_date has value = resolved
    """
    print("\n Loading conditions...")

    df = pd.read_csv(
        'output/csv/conditions.csv',
        dtype=str
    )

    print(f"   Found {len(df)} conditions")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            stop_date = clean_date(row['STOP'])
            is_active = stop_date is None

            cur.execute("""
                INSERT INTO conditions (
                    start_date, stop_date,
                    member_id, encounter_id,
                    system, code, description,
                    is_active
                ) VALUES (
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s
                )
            """, (
                clean_date(row['START']),
                stop_date,
                clean_uuid(row['PATIENT']),
                clean_uuid(row['ENCOUNTER']),
                clean_value(row['SYSTEM']),
                clean_value(row['CODE']),
                clean_value(row['DESCRIPTION']),
                is_active
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_medications():
    """
    Loads medications.csv → medications table

    All prescriptions — active and past.
    Provides medication context for
    clinical reasoning node.
    """
    print("\n Loading medications...")

    df = pd.read_csv(
        'output/csv/medications.csv',
        dtype=str
    )

    print(f"   Found {len(df)} medications")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO medications (
                    start_time, stop_time,
                    member_id, plan_id, encounter_id,
                    code, description,
                    base_cost, payer_coverage,
                    dispenses, total_cost,
                    reason_code, reason_description
                ) VALUES (
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s
                )
            """, (
                clean_datetime(row['START']),
                clean_datetime(row['STOP']),
                clean_uuid(row['PATIENT']),
                clean_uuid(row['PAYER']),
                clean_uuid(row['ENCOUNTER']),
                clean_value(row['CODE']),
                clean_value(row['DESCRIPTION']),
                clean_decimal(row['BASE_COST']),
                clean_decimal(row['PAYER_COVERAGE']),
                clean_integer(row['DISPENSES']),
                clean_decimal(row['TOTALCOST']),
                clean_value(row['REASONCODE']),
                clean_value(row['REASONDESCRIPTION'])
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


def load_imaging_studies():
    """
    Loads imaging_studies.csv → imaging_studies table

    X-rays, MRIs, CT scans done.
    Critical for MRI prior auth:
    "Has this patient had prior imaging?"
    "Was conservative treatment documented?"
    """
    print("\n Loading imaging_studies...")

    df = pd.read_csv(
        'output/csv/imaging_studies.csv',
        dtype=str
    )

    print(f"   Found {len(df)} imaging studies")

    conn = get_connection()
    cur = conn.cursor()

    success = 0
    failed = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO imaging_studies (
                    id, study_date,
                    member_id, encounter_id,
                    bodysite_code, bodysite_description,
                    modality_code, modality_description,
                    procedure_code
                ) VALUES (
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s
                )
                ON CONFLICT (id) DO NOTHING
            """, (
                clean_uuid(row['Id']),
                clean_datetime(row['DATE']),
                clean_uuid(row['PATIENT']),
                clean_uuid(row['ENCOUNTER']),
                clean_value(row['BODYSITE_CODE']),
                clean_value(row['BODYSITE_DESCRIPTION']),
                clean_value(row['MODALITY_CODE']),
                clean_value(row['MODALITY_DESCRIPTION']),
                clean_value(row['PROCEDURE_CODE'])
            ))
            success += 1
        except Exception as e:
            failed += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"   ✅ {success} inserted, {failed} failed")


# ── MAIN FUNCTION ─────────────────────────────────────
def main():
    """
    Runs all loading functions in correct order.
    
    Order matters because of foreign keys:
    → Can't insert member_insurance before members
    → Can't insert encounters before providers
    → etc.
    """
    print("=" * 50)
    print(" LOADING SYNTHEA DATA INTO SUPABASE")
    print("=" * 50)

    # Load in dependency order
    load_insurance_plans()   # no dependencies
    load_members()           # no dependencies
    load_providers()         # no dependencies
    load_member_insurance()  # needs members + plans
    load_encounters()        # needs members + providers + plans
    load_claims()            # needs members + providers + plans
    load_procedures()        # needs members + encounters
    load_conditions()        # needs members + encounters
    load_medications()       # needs members + plans + encounters
    load_imaging_studies()   # needs members + encounters

    print("\n" + "=" * 50)
    print(" ALL DATA LOADED SUCCESSFULLY")
    print("=" * 50)


# ── ENTRY POINT ───────────────────────────────────────
# This block runs only when you execute this file
# directly with: python3 data/load_data.py
#
# It does NOT run when another file imports from here
# That's what if __name__ == '__main__' means
if __name__ == '__main__':
    main()