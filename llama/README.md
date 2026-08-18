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

## PDF Size and Wait Time

The larger the PDF, the longer the user should expect to wait for processing and answers. Longer documents usually create more text chunks, so the application has more content to extract, embed, index, search, and process before generating a response.

Processing speed also depends on the computer running Ollama. CPU, RAM, GPU availability, model loading time, and the complexity of the document can all affect how long indexing and answer generation take. A short PDF may process relatively quickly, while a large PDF can take several minutes or longer on slower hardware.

## CPU vs GPU Performance and Ollama Troubleshooting

Ollama can run the Llama and embedding models on the CPU or, when supported and detected correctly, on a GPU. CPU-only execution can be significantly slower for RAG workloads because PDF indexing may require many embedding requests and answer generation must also run through the local model.

Check where the currently loaded model is running with:

```bash
ollama ps
```

The `PROCESSOR` column is the important field. For example:

```text
100% CPU
```

means the model is running entirely on the CPU, while:

```text
100% GPU
```

means the model is fully loaded on the GPU. A CPU/GPU split means Ollama has offloaded only part of the model to the GPU.

On systems with an NVIDIA GPU, verify that the operating system can see the GPU:

```bash
nvidia-smi
```

If `nvidia-smi` detects the GPU but `ollama ps` still reports `100% CPU`, check the Ollama service log:

```bash
journalctl -u ollama --no-pager -n 100
```

Messages such as `inference compute id=cpu`, `offloaded 0/... layers to GPU`, or `total vram="0 B"` indicate that Ollama did not successfully detect a usable GPU and has fallen back to CPU execution.

### Fix Used During Development on Debian/WSL2

During development, WSL2 could see the NVIDIA GPU through `nvidia-smi`, but an older Ollama installation still reported `100% CPU` and `0 B` of VRAM. Updating Ollama resolved GPU detection.

On Debian/Ubuntu, install `zstd` first if the current Ollama installer requires it:

```bash
sudo apt-get update
sudo apt-get install -y zstd
```

Then update/reinstall Ollama using the official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Restart the Ollama service:

```bash
sudo systemctl restart ollama
```

Run a quick model test:

```bash
ollama run llama3.2 "Reply with exactly: OK"
```

Then verify GPU use:

```bash
ollama ps
```

A successful fix should show the loaded model using the GPU, for example:

```text
PROCESSOR
100% GPU
```

You can also confirm active GPU memory and utilization with:

```bash
nvidia-smi
```

Do not assume that a long wait is caused by PDF size alone. If indexing finishes but answer generation is unexpectedly slow, check `ollama ps` and the Ollama logs to confirm whether the model is actually using the GPU.

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
- CPU-only Ollama execution can be substantially slower than GPU-accelerated execution.
- Larger PDFs generally require longer processing and answer times.
- Model responses can vary from the IBM Granite implementation.

## Development Status

This implementation is being developed on the `feature/llama-ollama` branch so it can be tested independently before being merged into `main`.
