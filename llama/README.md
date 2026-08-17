# Local Llama + Ollama RAG Implementation

This directory contains a fully local alternative to the IBM watsonx.ai implementation in the repository root.

It preserves the same core RAG workflow while replacing IBM-hosted embeddings and chat generation with local models served by **Ollama**.

## Stack

- Meta Llama through Ollama for answer generation
- `embeddinggemma` through Ollama for embeddings
- LangChain for orchestration
- Chroma for local vector storage and similarity retrieval
- pypdf for PDF text extraction
- Gradio for the user interface

## Models

Default models:

```text
Chat model: llama3.2
Embedding model: embeddinggemma
```

These values are configured in `rag_pipeline_llama.py`.

## Why This Version Exists

The original project demonstrates cloud AI integration through IBM watsonx.ai. This implementation demonstrates the same RAG architecture with a local model backend.

That provides two different deployment approaches in one portfolio project:

- cloud-hosted IBM Granite models through IBM watsonx.ai
- locally hosted Meta Llama and embeddings through Ollama

## Prerequisites

Install and run Ollama on the machine where the application will execute.

Then download the required models:

```bash
ollama pull llama3.2
ollama pull embeddinggemma
```

Verify the models are installed:

```bash
ollama list
```

The implementation expects Ollama at:

```text
http://127.0.0.1:11434
```

## Install Python Dependencies

From the repository root:

```bash
python3 -m venv .venv-llama
source .venv-llama/bin/activate
pip install -r llama/requirements-llama.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv-llama
.venv-llama\Scripts\Activate.ps1
pip install -r llama/requirements-llama.txt
```

## Run the Gradio Application

From the repository root:

```bash
python llama/app_llama.py
```

Then open the local Gradio URL shown in the terminal, normally:

```text
http://127.0.0.1:7860
```

## Run the Command-Line Pipeline Test

No third-party sample PDF is bundled with this implementation. Supply your own readable PDF:

```bash
python llama/rag_pipeline_llama.py /path/to/document.pdf "What is this document about?"
```

If the question is omitted, the default question is:

```text
What is this document about?
```

## RAG Workflow

```text
PDF Upload
    |
    v
pypdf Text Extraction
    |
    v
LangChain Text Chunking
    |
    v
Ollama embeddinggemma Embeddings
    |
    v
Chroma Vector Database
    |
    v
Semantic Retrieval
    |
    v
Retrieved PDF Context
    |
    v
Meta Llama through Ollama
    |
    v
Grounded Answer + Source Pages
```

## Behavior Preserved from the IBM Version

The local implementation keeps the major application behaviors already developed for the IBM version:

- PDF validation
- readable-page extraction
- overlapping text chunks
- batched embedding operations
- persistent Chroma storage
- replacement of the previous indexed document
- similarity retrieval
- grounded prompting
- source-page reporting
- Gradio application state
- disabled controls during active RAG operations
- serialized indexing and answer generation
- clear-chat behavior
- complete application reset
- user-friendly error handling

## Local Model Error Handling

If Ollama is stopped, unavailable, or a configured model is missing, the application displays a friendly Ollama availability message instead of exposing the full internal exception in the Gradio interface.

Technical details are still printed to the terminal for troubleshooting.

## Privacy and Cost Characteristics

With this implementation, PDF text, embeddings, retrieval, and answer generation stay on the local machine when Ollama is running locally.

The implementation does not require an IBM Cloud API key, IBM watsonx.ai project ID, or IBM token quota.

Local execution still requires sufficient CPU, RAM, and optionally GPU resources for the selected models.

## Files

```text
llama/
├── app_llama.py
├── rag_pipeline_llama.py
├── requirements-llama.txt
└── README.md
```

## Current Limitations

- Ollama must be installed and running separately.
- The configured models must be downloaded before use.
- PDF files must contain extractable text.
- Scanned image-only PDFs are not OCR processed.
- Only one PDF is indexed at a time.
- Local model performance depends on available hardware.
- Model responses can vary from the IBM Granite implementation.

## Development Status

This implementation is being developed on the `feature/llama-ollama` branch so it can be tested independently before being merged into `main`.
