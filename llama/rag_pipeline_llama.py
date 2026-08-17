import argparse
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# =========================================================
# Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# Configuration
# =========================================================

CHROMA_COLLECTION = "rag_pdf_documents_llama"

EMBEDDING_MODEL_ID = "embeddinggemma"
LLM_MODEL_ID = "llama3.2"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 8
DEFAULT_RETRIEVAL_K = 3

OLLAMA_ERROR_MESSAGE = (
    "Unable to connect to the local Ollama service or required model. "
    "Make sure Ollama is running and the configured models are installed."
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# PDF loading
# =========================================================

def document_loader(file_path: str):
    """Load readable PDF pages into LangChain Documents."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise ValueError("The PDF contains no pages.")

    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": page_number,
                    "page_label": str(page_number + 1),
                    "total_pages": total_pages,
                },
            )
        )

    if not documents:
        raise ValueError("No readable text was found in the PDF.")

    return documents


# =========================================================
# Text splitting
# =========================================================

def text_splitter(documents):
    """Split PDF text into overlapping chunks."""
    if not documents:
        raise ValueError("No documents were provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created.")

    return chunks


# =========================================================
# Ollama models
# =========================================================

def ollama_embedding():
    """Create the local Ollama embedding model."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL_ID,
        base_url=OLLAMA_BASE_URL,
    )


def get_llm():
    """Create the local Llama chat model through Ollama."""
    return ChatOllama(
        model=LLM_MODEL_ID,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=500,
    )


# =========================================================
# Chroma client and vector database
# =========================================================

def create_chroma_client():
    """Create the persistent local Chroma client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
        ),
    )


def vector_database(
    chunks,
    batch_size=DEFAULT_BATCH_SIZE,
    reset_db=True,
):
    """Embed document chunks locally and store them in Chroma."""
    if not chunks:
        raise ValueError("No document chunks were provided.")

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    client = create_chroma_client()

    if reset_db:
        logger.info("Resetting previous Llama Chroma database.")
        client.reset()

    vector_store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=ollama_embedding(),
    )

    total_chunks = len(chunks)

    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)
        batch = chunks[start:end]

        logger.info(
            "Embedding chunks %s-%s of %s with Ollama.",
            start + 1,
            end,
            total_chunks,
        )

        try:
            vector_store.add_documents(batch)
        except Exception as error:
            raise RuntimeError(OLLAMA_ERROR_MESSAGE) from error

    logger.info("Indexed %s chunks successfully.", total_chunks)
    return vector_store


# =========================================================
# Retriever
# =========================================================

def retriever(vector_store, k=DEFAULT_RETRIEVAL_K):
    """Create a similarity retriever from the Chroma vector store."""
    if vector_store is None:
        raise ValueError("A vector store must be provided.")

    if k < 1:
        raise ValueError("Retriever result count must be at least 1.")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# =========================================================
# Prompt
# =========================================================

def build_rag_prompt():
    """Build the grounded document Q&A prompt."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document question-answering assistant.

Answer the user's question using only the retrieved document context provided to you.
Do not use outside knowledge.
Do not invent, assume, or add information that is not supported by the supplied context.

If the context does not contain enough information to answer the question, respond exactly with:

I could not find enough information in the document to answer that question.

Be clear, concise, and factual.""",
            ),
            (
                "human",
                """Retrieved document context:

{context}

Question:

{question}""",
            ),
        ]
    )


# =========================================================
# RAG answer generation
# =========================================================

def rag_answer(document_retriever, query: str):
    """Retrieve relevant chunks and answer with local Llama."""
    if document_retriever is None:
        raise ValueError("A document retriever must be provided.")

    if not query or not query.strip():
        raise ValueError("A question must be provided.")

    query = query.strip()

    try:
        documents = document_retriever.invoke(query)
    except Exception as error:
        raise RuntimeError(OLLAMA_ERROR_MESSAGE) from error

    if not documents:
        return (
            "I could not find enough information in the document to answer that question.",
            [],
        )

    context_sections = []

    for document in documents:
        page_number = document.metadata.get("page", 0) + 1
        context_sections.append(
            f"[Page {page_number}]\n{document.page_content}"
        )

    context = "\n\n".join(context_sections)
    chain = build_rag_prompt() | get_llm()

    try:
        response = chain.invoke(
            {
                "context": context,
                "question": query,
            }
        )
    except Exception as error:
        raise RuntimeError(OLLAMA_ERROR_MESSAGE) from error

    return response.content, documents


# =========================================================
# Source formatting
# =========================================================

def format_sources(source_documents):
    """Convert retrieved metadata into readable source citations."""
    sources = []
    seen_sources = set()

    for document in source_documents:
        source = (
            document.metadata.get("filename")
            or document.metadata.get("source")
            or "Unknown source"
        )
        page_number = document.metadata.get("page", 0) + 1
        source_key = (source, page_number)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        sources.append(f"{source} (page {page_number})")

    return sources


# =========================================================
# Complete indexing workflow
# =========================================================

def index_pdf(
    file_path: str,
    batch_size=DEFAULT_BATCH_SIZE,
    k=DEFAULT_RETRIEVAL_K,
):
    """Load, split, embed, index, and prepare a PDF retriever."""
    path = Path(file_path)

    logger.info("Loading PDF: %s", path.name)
    documents = document_loader(str(path))
    logger.info("Loaded %s readable pages.", len(documents))

    chunks = text_splitter(documents)
    logger.info("Created %s chunks.", len(chunks))

    vector_store = vector_database(
        chunks,
        batch_size=batch_size,
        reset_db=True,
    )

    document_retriever = retriever(vector_store, k=k)

    return document_retriever, len(documents), len(chunks)


# =========================================================
# Command-line test
# =========================================================

def run_pipeline_test(pdf_path: str, query: str):
    """Run an end-to-end local test using a user-supplied PDF."""
    print("=" * 60)
    print("Local Llama/Ollama RAG Pipeline Test")
    print("=" * 60)

    document_retriever, page_count, chunk_count = index_pdf(pdf_path)

    print(f"\nReadable pages: {page_count}")
    print(f"Chunks indexed: {chunk_count}")
    print("\nQUESTION")
    print(query)

    answer, source_documents = rag_answer(document_retriever, query)

    print("\nGENERATED ANSWER")
    print(answer)
    print("\nRETRIEVED SOURCES")

    sources = format_sources(source_documents)
    if sources:
        for source in sources:
            print(f"- {source}")
    else:
        print("No source documents were retrieved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the local Llama/Ollama RAG pipeline with a PDF."
    )
    parser.add_argument("pdf", help="Path to a readable PDF file")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is this document about?",
        help="Question to ask about the PDF",
    )
    args = parser.parse_args()

    try:
        run_pipeline_test(args.pdf, args.question)
    except Exception as error:
        logger.exception("Pipeline test failed.")
        print(f"\nPIPELINE ERROR\n{error}")
