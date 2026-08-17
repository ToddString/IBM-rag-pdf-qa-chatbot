# Retrieval-Augmented Generation (RAG) PDF Question Answering Chatbot

A portfolio-ready PDF question-answering application built with **Retrieval-Augmented Generation (RAG)**. Users can upload a PDF, index its contents, and ask grounded questions whose answers are generated from retrieved document context.

The application uses **IBM watsonx.ai** for embeddings and answer generation, **LangChain** for orchestration, **Chroma** for vector storage and similarity search, **pypdf** for PDF text extraction, and **Gradio** for the user interface.

## Features

- Upload PDF documents through a Gradio interface
- Extract readable text from PDF pages with pypdf
- Split document text into overlapping chunks
- Generate embeddings with IBM watsonx.ai
- Store document embeddings in a persistent Chroma vector database
- Retrieve relevant passages using semantic similarity
- Generate grounded answers with an IBM Granite chat model
- Display source pages used for each answer
- Replace the previously indexed PDF when a new document is indexed
- Disable conflicting controls during indexing and answer generation
- Serialize resource-intensive RAG operations to avoid race conditions
- Handle IBM watsonx.ai token quota errors gracefully
- Clear chat history without removing the indexed document
- Reset the complete application state

## Technology Stack

- Python 3.11
- IBM watsonx.ai
- IBM Granite models
- LangChain
- Chroma
- Gradio
- pypdf
- python-dotenv

## Models

The project currently uses:

- **Embedding model:** `ibm/granite-embedding-278m-multilingual`
- **Chat model:** `ibm/granite-4-h-small`

## How It Works

The application follows this RAG workflow:

```text
PDF Upload
    |
    v
PDF Text Extraction
    |
    v
Text Chunking
    |
    v
IBM watsonx.ai Embeddings
    |
    v
Chroma Vector Database
    |
    v
Semantic Retrieval
    |
    v
Relevant Document Context
    |
    v
IBM Granite Chat Model
    |
    v
Grounded Answer + Source Pages
```

When a user uploads and indexes a PDF:

1. pypdf extracts text from readable PDF pages.
2. LangChain splits the extracted text into overlapping chunks.
3. IBM watsonx.ai generates vector embeddings for the chunks.
4. Chroma stores the embeddings and document metadata.
5. A LangChain retriever performs similarity search when the user asks a question.
6. The most relevant document chunks are passed to the IBM Granite chat model.
7. The model generates an answer using only the retrieved document context.
8. The application displays the source pages associated with the retrieved chunks.

## Project Structure

```text
rag-pdf-qa-chatbot/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── sample_docs/
├── screenshots/
│   ├── app-home.png
│   ├── pdf-indexed.png
│   └── question-answer.png
└── tests/
    └── README.md
```

Generated and sensitive files such as `.env`, `.venv`, `chroma_db`, and Python cache files are excluded through `.gitignore`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ToddString/IBM-rag-pdf-qa-chatbot.git
cd IBM-rag-pdf-qa-chatbot
```

### 2. Create a virtual environment

Linux, WSL, or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## IBM watsonx.ai Configuration

This application requires access to an IBM watsonx.ai project and an IBM Cloud API key.

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
WATSONX_APIKEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> **Security:** Do not commit your real `.env` file or IBM Cloud API key to GitHub.

## Running the Pipeline Test

A command-line end-to-end test is included in `rag_pipeline.py`.

Run:

```bash
python rag_pipeline.py
```

The test:

- loads the sample PDF
- extracts readable pages
- creates text chunks
- generates embeddings
- indexes the chunks in Chroma
- performs semantic retrieval
- generates a grounded answer
- displays retrieved source pages

Example output:

```text
============================================================
RAG PDF QA Chatbot - Pipeline Test
============================================================

Readable pages: 11
Chunks indexed: 38

QUESTION
What is Low-Rank Adaptation?

GENERATED ANSWER
...

RETRIEVED SOURCES
Source 1: test.pdf (page 2)
Source 2: test.pdf (page 10)
Source 3: test.pdf (page 1)
```

## Running the Gradio Application

Start the application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Using the Application

1. Upload a PDF document.
2. Click **Index PDF**.
3. Wait for indexing to finish.
4. Enter a question about the document.
5. Click **Ask** or press Enter.
6. Review the generated answer and source pages.

A newly indexed PDF replaces the previously indexed document.

## Screenshots

### Application Interface

![Application interface](screenshots/app-home.png)

### PDF Successfully Indexed

![PDF indexed](screenshots/pdf-indexed.png)

### Question Answering with Sources

![Question answering with sources](screenshots/question-answer.png)

## Error Handling

The application includes user-friendly handling for common failure conditions, including:

- missing PDF files
- unsupported file types
- PDFs containing no readable text
- missing IBM watsonx.ai environment variables
- IBM API failures
- IBM watsonx.ai token quota exhaustion
- attempts to ask questions before a PDF is indexed
- blank questions

If IBM watsonx.ai reports that the available token quota has been reached, the Gradio interface displays a specific quota warning instead of exposing the full API traceback to the user.

Technical error details are still written to the terminal for troubleshooting.

## Concurrency and Application State

Indexing and answer generation are serialized through the Gradio event queue.

Interactive controls are temporarily disabled while resource-intensive RAG operations are running. This prevents conflicting actions such as asking a question while a document is still being indexed or resetting the application during an active request.

## Vector Database Behavior

Chroma is configured as a persistent local vector database.

When a new document is indexed, the previous vector database contents are safely reset before the new document is stored. The application currently operates as a single-document question-answering system rather than a multi-document knowledge base.

The local Chroma database directory is excluded from Git.

## Retrieval Configuration

Current defaults:

- **Chunk size:** 1000 characters
- **Chunk overlap:** 200 characters
- **Embedding batch size:** 8 chunks
- **Retrieved chunks per question:** 3

These settings can be changed in `rag_pipeline.py`.

## Security

The project uses environment variables for IBM watsonx.ai credentials.

The following files and directories should never be committed:

```text
.env
.venv/
chroma_db/
__pycache__/
```

The included `.env.example` contains placeholders only.

Before publishing changes, verify ignored files with:

```bash
git status --short --ignored
```

## Current Limitations

- PDF files must contain extractable text
- Scanned image-only PDFs are not OCR processed
- Only one PDF is indexed at a time
- Large PDFs may consume significant IBM watsonx.ai token usage
- Answer quality depends on retrieval quality and the contents of the source document
- The application is currently intended for local use and portfolio demonstration

## Dependencies

The tested environment uses:

```text
chromadb==1.5.9
gradio==6.24.0
ibm-watsonx-ai==1.6.3
langchain-chroma==1.1.0
langchain-core==1.5.5
langchain-ibm==1.1.0
langchain-text-splitters==1.1.2
pypdf==6.16.1
python-dotenv==1.2.2
```

## Project Background

This project began as an **IBM Skills Network learning exercise** covering document loading, text splitting, embeddings, vector databases, retrieval, large language models, and Gradio.

The original course implementation was rebuilt and expanded into this standalone portfolio project with updated IBM watsonx.ai and LangChain integrations, modern Chroma usage, persistent vector storage, batched embeddings, source-page reporting, application-state management, concurrency protection, improved error handling, and a redesigned Gradio interface.

## License

No license has been selected yet.

Before redistributing third-party sample documents, verify that their licenses permit inclusion in this repository.
