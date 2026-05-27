import azure.functions as func
import logging
import uuid
import json
import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from openai import AzureOpenAI, OpenAI

app = func.FunctionApp()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

def get_cosmos_container():
    client = CosmosClient(
        os.environ["COSMOS_ENDPOINT"],
        os.environ["COSMOS_KEY"]
    )
    return client \
        .get_database_client(os.environ["COSMOS_DATABASE"]) \
        .get_container_client(os.environ["COSMOS_CONTAINER"])


def get_history_container():
    client = CosmosClient(
        os.environ["COSMOS_ENDPOINT"],
        os.environ["COSMOS_KEY"]
    )
    return client \
        .get_database_client(os.environ["COSMOS_DATABASE_2"]) \
        .get_container_client(os.environ["COSMOS_CONTAINER_HISTORY"])


def get_openai_client():
    return AzureOpenAI(
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key        = os.environ["AZURE_OPENAI_KEY"],
        api_version    = "2024-02-01"
    )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def generate_embedding(openai_client, text: str):
    response = openai_client.embeddings.create(
        input = text,
        model = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    )
    return response.data[0].embedding


# ══════════════════════════════════════════════════════
# FUNCTION 1 — UPLOAD PDF (HTTP trigger from UI)
# Called when user uploads PDF from website
# ══════════════════════════════════════════════════════

@app.route(
    route      = "upload",
    methods    = ["POST"],
    auth_level = func.AuthLevel.ANONYMOUS
)
def upload_pdf(req: func.HttpRequest) -> func.HttpResponse:
    """
    Receives PDF from Web UI
    Saves to row-data Blob container
    Then processes it automatically
    """
    logger.info("📤 PDF Upload request received")

    try:
        # Get file from request
        files = req.files.get("file")
        if not files:
            return func.HttpResponse(
                json.dumps({"error": "No file uploaded"}),
                mimetype    = "application/json",
                status_code = 400
            )

        file_name  = files.filename
        file_bytes = files.read()

        if not file_name.lower().endswith(".pdf"):
            return func.HttpResponse(
                json.dumps({"error": "Only PDF files allowed"}),
                mimetype    = "application/json",
                status_code = 400
            )

        logger.info(f"Processing file: {file_name}")

        # ── Step 1: Save PDF to Blob Storage ──────────────────
        blob_service = BlobServiceClient.from_connection_string(
            os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        )
        blob_client = blob_service.get_blob_client("row-data", file_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        logger.info(f"✅ PDF saved to row-data/{file_name}")

        # ── Step 2: Extract text ───────────────────────────────
        doc_client = DocumentIntelligenceClient(
            os.environ["AZURE_DOC_INTEL_ENDPOINT"],
            AzureKeyCredential(os.environ["AZURE_DOC_INTEL_KEY"])
        )
        poller = doc_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=file_bytes)
        )
        result = poller.result()

        extracted_pages = []
        for page_num, page in enumerate(result.pages, start=1):
            lines     = [line.content for line in page.lines]
            page_text = f"--- Page {page_num} ---\n" + "\n".join(lines)
            extracted_pages.append(page_text)
        plain_text = "\n\n".join(extracted_pages)
        logger.info(f"✅ Extracted {len(plain_text)} characters")

        # ── Step 3: Save text to processed-data ───────────────
        txt_name   = file_name.replace(".pdf", "_parsed.txt")
        txt_blob   = blob_service.get_blob_client("processed-data", txt_name)
        txt_blob.upload_blob(plain_text, overwrite=True)
        logger.info(f"✅ Text saved to processed-data/{txt_name}")

        # ── Step 4: Chunk text ─────────────────────────────────
        chunks = chunk_text(plain_text, chunk_size=500, overlap=50)
        logger.info(f"✅ Created {len(chunks)} chunks")

        # ── Step 5: Generate embeddings + store in Cosmos DB ───
        container     = get_cosmos_container()
        openai_client = get_openai_client()

        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(openai_client, chunk)
            document  = {
                "id"         : str(uuid.uuid4()),
                "content"    : chunk,
                "chunk_index": i,
                "embedding"  : embedding,
                "metadata"   : {
                    "source"   : file_name,
                    "section"  : f"chunk_{i}",
                    "timestamp": "2026-05-27"
                }
            }
            container.upsert_item(document)
            logger.info(f"✅ Stored chunk {i+1}/{len(chunks)}")

        logger.info(f"🎉 Pipeline complete for {file_name}!")

        return func.HttpResponse(
            json.dumps({
                "success" : True,
                "message" : f"✅ {file_name} processed successfully!",
                "chunks"  : len(chunks)
            }),
            mimetype = "application/json"
        )

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype    = "application/json",
            status_code = 500
        )


# ══════════════════════════════════════════════════════
# FUNCTION 2 — RAG CHAT (HTTP trigger from UI)
# Called when user asks a question on website
# ══════════════════════════════════════════════════════

@app.route(
    route      = "chat",
    methods    = ["POST"],
    auth_level = func.AuthLevel.ANONYMOUS
)
def rag_chat(req: func.HttpRequest) -> func.HttpResponse:
    """
    Receives question from Web UI
    Searches Cosmos DB for relevant chunks
    Returns AI generated answer
    """
    logger.info("💬 Chat request received")

    try:
        req_body   = req.get_json()
        question   = req_body.get("question", "").strip()
        session_id = req_body.get("session_id", str(uuid.uuid4()))

        if not question:
            return func.HttpResponse(
                json.dumps({"error": "No question provided"}),
                mimetype    = "application/json",
                status_code = 400
            )

        logger.info(f"Question: {question}")

        # ── Step 1: Convert question to embedding ──────────────
        openai_client   = get_openai_client()
        query_embedding = generate_embedding(openai_client, question)

        # ── Step 2: Vector search in Cosmos DB ────────────────
        container    = get_cosmos_container()
        search_query = """
        SELECT TOP 3
            c.content,
            c.chunk_index,
            c.metadata,
            VectorDistance(c.embedding, @embedding) AS score
        FROM c
        ORDER BY VectorDistance(c.embedding, @embedding)
        """
        results = list(container.query_items(
            query      = search_query,
            parameters = [{"name": "@embedding", "value": query_embedding}],
            enable_cross_partition_query = True
        ))

        if not results:
            return func.HttpResponse(
                json.dumps({"answer": "No relevant documents found. Please upload a PDF first."}),
                mimetype = "application/json"
            )

        logger.info(f"✅ Found {len(results)} relevant chunks")

        # ── Step 3: Generate answer using GPT ─────────────────
        chat_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        context     = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['content']}"
            for i, chunk in enumerate(results)
        ])

        response = chat_client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [
                {
                    "role"   : "system",
                    "content": """You are a helpful assistant.
                    Answer questions based ONLY on the context provided.
                    Be detailed and specific.
                    If answer not found say:
                    'I cannot find this in the uploaded documents.'"""
                },
                {
                    "role"   : "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            max_tokens  = 500,
            temperature = 0.3
        )
        answer = response.choices[0].message.content
        logger.info("✅ Answer generated")

        # ── Step 4: Save conversation history ─────────────────
        try:
            history = get_history_container()
            history.upsert_item({
                "id"        : str(uuid.uuid4()),
                "session_id": session_id,
                "query"     : question,
                "response"  : answer,
                "timestamp" : "2026-05-27"
            })
        except Exception as e:
            logger.warning(f"History save failed: {e}")

        return func.HttpResponse(
            json.dumps({"answer": answer}),
            mimetype = "application/json"
        )

    except Exception as e:
        logger.error(f"❌ Chat failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype    = "application/json",
            status_code = 500
        )


# ══════════════════════════════════════════════════════
# FUNCTION 3 — LIST UPLOADED DOCUMENTS
# Shows what PDFs have been uploaded
# ══════════════════════════════════════════════════════

@app.route(
    route      = "documents",
    methods    = ["GET"],
    auth_level = func.AuthLevel.ANONYMOUS
)
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    """Returns list of uploaded PDF files"""
    try:
        blob_service = BlobServiceClient.from_connection_string(
            os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        )
        container = blob_service.get_container_client("row-data")
        blobs     = [
            blob.name
            for blob in container.list_blobs()
            if blob.name.lower().endswith(".pdf")
        ]
        return func.HttpResponse(
            json.dumps({"documents": blobs}),
            mimetype = "application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype    = "application/json",
            status_code = 500
        )
