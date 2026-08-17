import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# =========================================================
# Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# Configuration
# =========================================================

load_dotenv(ENV_FILE)

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

CHROMA_COLLECTION = "rag_pdf_documents"

EMBEDDING_MODEL_ID = (
    "ibm/granite-embedding-278m-multilingual"
)

LLM_MODEL_ID = "ibm/granite-4-h-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_RETRIEVAL_K = 3

QUOTA_ERROR_MESSAGE = (
    "IBM watsonx.ai token quota has been reached. "
    "The request cannot be completed until additional "
    "token capacity is available."
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# =========================================================
# Environment validation
# =========================================================

def validate_environment():
    """
    Verify that all required IBM watsonx environment
    variables are configured.
    """
    required_variables = {
        "WATSONX_APIKEY": WATSONX_APIKEY,
        "WATSONX_PROJECT_ID": WATSONX_PROJECT_ID,
        "WATSONX_URL": WATSONX_URL,
    }

    missing_variables = [
        name
        for name, value in required_variables.items()
        if not value
    ]

    if missing_variables:
        missing = ", ".join(
            missing_variables
        )

        raise ValueError(
            "Missing required environment variable(s): "
            f"{missing}"
        )


# =========================================================
# IBM API error handling
# =========================================================

def handle_watsonx_error(error):
    """
    Convert known IBM watsonx API errors into cleaner
    application-level error messages.
    """
    error_text = str(error)

    if (
        "token_quota_reached" in error_text
        or "Token consumption quota has been reached"
        in error_text
    ):
        raise RuntimeError(
            QUOTA_ERROR_MESSAGE
        ) from error

    raise error


# =========================================================
# PDF loading
# =========================================================

def document_loader(
    file_path: str,
):
    """
    Load a PDF with pypdf and convert each readable page
    into a LangChain Document object.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    reader = PdfReader(
        str(path)
    )

    total_pages = len(
        reader.pages
    )

    if total_pages == 0:
        raise ValueError(
            "The PDF contains no pages."
        )

    documents = []

    for page_number, page in enumerate(
        reader.pages
    ):
        text = (
            page.extract_text()
            or ""
        )

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": page_number,
                    "page_label": str(
                        page_number + 1
                    ),
                    "total_pages": total_pages,
                },
            )
        )

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    return documents


# =========================================================
# Text splitting
# =========================================================

def text_splitter(
    documents,
):
    """
    Split readable PDF pages into overlapping chunks
    suitable for embedding and semantic retrieval.
    """
    if not documents:
        raise ValueError(
            "No documents were provided for splitting."
        )

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    return chunks


# =========================================================
# IBM watsonx embeddings
# =========================================================

def watsonx_embedding():
    """
    Create the IBM watsonx embedding model.
    """
    validate_environment()

    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        apikey=WATSONX_APIKEY,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
    )


# =========================================================
# IBM watsonx chat model
# =========================================================

def get_llm():
    """
    Create the IBM watsonx chat model used for grounded
    answer generation.
    """
    validate_environment()

    return ChatWatsonx(
        model_id=LLM_MODEL_ID,
        apikey=WATSONX_APIKEY,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        params={
            "temperature": 0.1,
            "max_tokens": 500,
        },
    )


# =========================================================
# Chroma client
# =========================================================

def create_chroma_client():
    """
    Create the persistent local Chroma client.
    """
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
        ),
    )


# =========================================================
# Chroma vector database
# =========================================================

def vector_database(
    chunks,
    batch_size=DEFAULT_BATCH_SIZE,
    reset_db=True,
):
    """
    Build a Chroma vector store from document chunks.

    When reset_db=True, the existing Chroma database is
    safely reset before the new document is indexed.
    """
    if not chunks:
        raise ValueError(
            "No document chunks were provided."
        )

    if batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    client = create_chroma_client()

    if reset_db:
        logger.info(
            "Resetting previous Chroma database."
        )

        client.reset()

    embedding_model = (
        watsonx_embedding()
    )

    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )

    total_chunks = len(
        chunks
    )

    for start in range(
        0,
        total_chunks,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total_chunks,
        )

        batch = chunks[
            start:end
        ]

        logger.info(
            "Embedding chunks %s-%s of %s.",
            start + 1,
            end,
            total_chunks,
        )

        try:
            vector_store.add_documents(
                batch
            )

        except ApiRequestFailure as error:
            handle_watsonx_error(
                error
            )

    logger.info(
        "Indexed %s chunks successfully.",
        total_chunks,
    )

    return vector_store


# =========================================================
# Retriever
# =========================================================

def retriever(
    vector_store,
    k=DEFAULT_RETRIEVAL_K,
):
    """
    Convert a Chroma vector store into a similarity
    retriever.
    """
    if vector_store is None:
        raise ValueError(
            "A vector store must be provided."
        )

    if k < 1:
        raise ValueError(
            "Retriever result count must be at least 1."
        )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )


# =========================================================
# Prompt
# =========================================================

def build_rag_prompt():
    """
    Build the prompt used for grounded document Q&A.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a document question-answering assistant.

Answer the user's question using only the retrieved
document context provided to you.

Do not use outside knowledge.

Do not invent, assume, or add information that is not
supported by the supplied document context.

If the document context does not contain enough
information to answer the question, respond exactly with:

I could not find enough information in the document to answer that question.

Be clear, concise, and factual.
""",
            ),
            (
                "human",
                """
Retrieved document context:

{context}

Question:

{question}
""",
            ),
        ]
    )


# =========================================================
# RAG answer generation
# =========================================================

def rag_answer(
    document_retriever,
    query: str,
):
    """
    Retrieve relevant chunks and generate a grounded
    answer using IBM watsonx.
    """
    if document_retriever is None:
        raise ValueError(
            "A document retriever must be provided."
        )

    if not query:
        raise ValueError(
            "A question must be provided."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "A question must be provided."
        )

    try:
        documents = (
            document_retriever.invoke(
                query
            )
        )

    except ApiRequestFailure as error:
        handle_watsonx_error(
            error
        )

    if not documents:
        return (
            "I could not find enough information in the "
            "document to answer that question.",
            [],
        )

    context_sections = []

    for document in documents:
        page_number = (
            document.metadata.get(
                "page",
                0,
            )
            + 1
        )

        context_sections.append(
            (
                f"[Page {page_number}]\n"
                f"{document.page_content}"
            )
        )

    context = "\n\n".join(
        context_sections
    )

    prompt = build_rag_prompt()
    chat_model = get_llm()

    chain = (
        prompt
        | chat_model
    )

    try:
        response = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )

    except ApiRequestFailure as error:
        handle_watsonx_error(
            error
        )

    answer = response.content

    return (
        answer,
        documents,
    )


# =========================================================
# Source formatting
# =========================================================

def format_sources(
    source_documents,
):
    """
    Convert retrieved document metadata into readable
    source citations.
    """
    sources = []
    seen_sources = set()

    for document in source_documents:
        source = (
            document.metadata.get(
                "filename"
            )
            or document.metadata.get(
                "source"
            )
            or "Unknown source"
        )

        page_number = (
            document.metadata.get(
                "page",
                0,
            )
            + 1
        )

        source_key = (
            source,
            page_number,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(
            source_key
        )

        sources.append(
            f"{source} "
            f"(page {page_number})"
        )

    return sources


# =========================================================
# Complete indexing workflow
# =========================================================

def index_pdf(
    file_path: str,
    batch_size=DEFAULT_BATCH_SIZE,
    k=DEFAULT_RETRIEVAL_K,
):
    """
    Complete PDF indexing workflow.

    The previous Chroma database is reset, then the new
    PDF is loaded, split, embedded, indexed, and converted
    into a retriever.
    """
    path = Path(
        file_path
    )

    logger.info(
        "Loading PDF: %s",
        path.name,
    )

    documents = document_loader(
        str(path)
    )

    logger.info(
        "Loaded %s readable pages.",
        len(documents),
    )

    chunks = text_splitter(
        documents
    )

    logger.info(
        "Created %s chunks.",
        len(chunks),
    )

    vector_store = vector_database(
        chunks,
        batch_size=batch_size,
        reset_db=True,
    )

    document_retriever = retriever(
        vector_store,
        k=k,
    )

    return (
        document_retriever,
        len(documents),
        len(chunks),
    )


# =========================================================
# Local pipeline test
# =========================================================

def run_pipeline_test():
    """
    Run a local end-to-end test with the sample PDF.
    """
    pdf_path = (
        BASE_DIR
        / "sample_docs"
        / "test.pdf"
    )

    query = (
        "What is Low-Rank Adaptation?"
    )

    print("=" * 60)
    print("RAG PDF QA Chatbot - Pipeline Test")
    print("=" * 60)

    try:
        (
            document_retriever,
            page_count,
            chunk_count,
        ) = index_pdf(
            str(pdf_path)
        )

        print()
        print("=" * 60)
        print("QUESTION")
        print("=" * 60)
        print(query)

        answer, source_documents = (
            rag_answer(
                document_retriever,
                query,
            )
        )

        print()
        print("=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)
        print(answer)

        print()
        print("=" * 60)
        print("RETRIEVED SOURCES")
        print("=" * 60)

        sources = format_sources(
            source_documents
        )

        if not sources:
            print(
                "No source documents were retrieved."
            )
        else:
            for index, source in enumerate(
                sources,
                start=1,
            ):
                print(
                    f"Source {index}: {source}"
                )

        print()
        print("=" * 60)
        print("INDEX SUMMARY")
        print("=" * 60)

        print(
            f"Readable pages: {page_count}"
        )

        print(
            f"Chunks indexed: {chunk_count}"
        )

        print()
        print(
            "Pipeline test complete."
        )

    except RuntimeError as error:
        print()
        print("=" * 60)
        print("PIPELINE ERROR")
        print("=" * 60)

        print(
            str(error)
        )


# =========================================================
# Application entry point
# =========================================================

if __name__ == "__main__":
    run_pipeline_test()