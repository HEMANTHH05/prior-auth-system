# data/index_policies.py
# ─────────────────────────────────────────────────────
# PURPOSE: Index policy documents into Azure AI Search
# RUN ONCE: python3 data/index_policies.py
# ─────────────────────────────────────────────────────

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

load_dotenv()

# ── CLIENTS ───────────────────────────────────────────
index_client = SearchIndexClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
)

search_client = SearchClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    index_name=os.getenv('AZURE_SEARCH_INDEX'),
    credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
)

openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


# ── STEP 1: CREATE SEARCH INDEX ───────────────────────
def create_search_index():
    """
    Creates the Azure AI Search index schema.

    Fields:
    → id: unique key per chunk
    → content: actual policy text
    → filename: source file name
    → plan_name: which insurance plan
    → procedure_type: mri, pt, surgery etc
    → content_vector: embedding for semantic search
    """
    print("\nCreating Azure AI Search index...")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SimpleField(
            name="filename",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="plan_name",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="procedure_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )
        ]
    )

    index = SearchIndex(
        name=os.getenv('AZURE_SEARCH_INDEX'),
        fields=fields,
        vector_search=vector_search
    )

    # Delete existing index if it exists
    try:
        index_client.delete_index(
            os.getenv('AZURE_SEARCH_INDEX')
        )
        print("   Deleted existing index")
    except:
        pass

    index_client.create_index(index)
    print("   ✅ Index created successfully")


# ── STEP 2: CHUNK DOCUMENTS ───────────────────────────
def chunk_document(content, chunk_size=500, overlap=50):
    """
    Splits large documents into smaller chunks.

    chunk_size=500: ~500 chars per chunk
    overlap=50: chunks share 50 chars at boundaries

    Why overlap?
    → Prevents cutting sentences mid-thought
    → Maintains context between adjacent chunks
    """
    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size

        if end < len(content):
            last_space = content.rfind(' ', start, end)
            if last_space > start:
                end = last_space

        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ── STEP 3: GENERATE EMBEDDING ────────────────────────
def get_embedding(text):
    """
    Converts text to vector using OpenAI embeddings.

    Model: text-embedding-3-small
    Output: list of 1536 numbers
    These numbers represent semantic meaning.
    Similar text = similar numbers = found by RAG.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ── STEP 4: EXTRACT METADATA ──────────────────────────
def extract_metadata(filename):
    """
    Determines plan_name and procedure_type
    from the filename.

    Used as filters in policy_retrieval_node:
    "only return policies matching this plan"
    """
    name = filename.replace('.txt', '').replace('.pdf', '')

    # Determine plan name
    if 'medicare' in name:
        plan_name = 'medicare'
    elif 'medicaid' in name:
        plan_name = 'medicaid'
    elif 'aetna' in name:
        plan_name = 'private'
    elif 'uhc' in name or 'united' in name:
        plan_name = 'private'
    elif 'cigna' in name:
        plan_name = 'private'
    elif 'private' in name:
        plan_name = 'private'
    elif 'all' in name:
        plan_name = 'all'
    else:
        plan_name = 'unknown'

    # Determine procedure type
    if 'mri' in name:
        procedure_type = 'mri'
    elif 'physical_therapy' in name or 'pt' in name:
        procedure_type = 'physical_therapy'
    elif 'surgery' in name or 'surgical' in name:
        procedure_type = 'surgery'
    elif 'specialist' in name:
        procedure_type = 'specialist_referral'
    elif 'cardiac' in name or 'stress' in name:
        procedure_type = 'cardiac'
    elif 'mental_health' in name:
        procedure_type = 'mental_health'
    elif 'diabetes' in name:
        procedure_type = 'diabetes'
    else:
        procedure_type = 'general'

    return plan_name, procedure_type


# ── STEP 5: READ FILE CONTENT ─────────────────────────
def read_file(filepath, filename):
    """
    Reads content from both PDF and TXT files.

    PDF: uses PyMuPDF to extract text page by page
    TXT: standard file read

    Returns text content as string.
    """
    if filename.endswith('.pdf'):
        content = ""
        try:
            doc = fitz.open(filepath)
            for page in doc:
                content += page.get_text()
            doc.close()

            if not content.strip():
                print(f"   ⚠️  No text extracted from {filename}")
                return None
        except Exception as e:
            print(f"   ❌ Failed to read PDF {filename}: {e}")
            return None
        return content

    elif filename.endswith('.txt'):
        with open(filepath, 'r') as f:
            return f.read()

    return None


# ── STEP 6: INDEX ALL DOCUMENTS ───────────────────────
def index_all_policies():
    """
    Main indexing function:
    1. Reads every .txt and .pdf in data/policies/
    2. Extracts text content
    3. Chunks into 500-char pieces
    4. Generates OpenAI embedding per chunk
    5. Uploads all to Azure AI Search
    """
    policies_dir = 'data/policies'
    documents_to_upload = []
    chunk_count = 0
    file_count = 0

    print("\nIndexing policy documents...")
    print("=" * 50)

    for filename in sorted(os.listdir(policies_dir)):

        # Skip non-policy files
        if not (filename.endswith('.txt') or
                filename.endswith('.pdf')):
            continue

        # Skip the create_policies.py script
        if filename.endswith('.py'):
            continue

        filepath = os.path.join(policies_dir, filename)

        # Read file content
        content = read_file(filepath, filename)
        if not content:
            continue

        # Extract plan and procedure metadata
        plan_name, procedure_type = extract_metadata(filename)

        # Chunk the document
        chunks = chunk_document(content)

        file_count += 1
        print(f"\n📄 {filename}")
        print(f"   Plan: {plan_name} | "
              f"Procedure: {procedure_type} | "
              f"Chunks: {len(chunks)}")

        # Generate embedding for each chunk
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)

                doc = {
                    "id": f"{filename.replace('.', '_')}_{i}",
                    "content": chunk,
                    "filename": filename,
                    "plan_name": plan_name,
                    "procedure_type": procedure_type,
                    "content_vector": embedding
                }

                documents_to_upload.append(doc)
                chunk_count += 1

            except Exception as e:
                print(f"   ❌ Chunk {i} failed: {e}")

    # Upload in batches of 100
    print(f"\n⬆️  Uploading {chunk_count} chunks "
          f"from {file_count} files...")

    batch_size = 100
    total_batches = (len(documents_to_upload) +
                     batch_size - 1) // batch_size

    for i in range(0, len(documents_to_upload), batch_size):
        batch = documents_to_upload[i:i + batch_size]
        search_client.upload_documents(documents=batch)
        current_batch = i // batch_size + 1
        print(f"   Uploaded batch "
              f"{current_batch}/{total_batches}")

    print("\n" + "=" * 50)
    print(f"✅ Indexed {chunk_count} chunks "
          f"from {file_count} documents")
    print("✅ Azure AI Search ready for RAG queries")


# ── MAIN ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print(" INDEXING POLICIES TO AZURE AI SEARCH")
    print("=" * 50)

    create_search_index()
    index_all_policies()

    print("\n✅ Done! RAG pipeline ready.")


if __name__ == '__main__':
    main()