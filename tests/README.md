# Testing Guide

This directory documents the manual validation performed for the **RAG PDF Question Answering Chatbot**.

The repository contains two implementations:

- IBM watsonx.ai + IBM Granite
- Local Meta Llama + Ollama

The project currently uses manual end-to-end testing rather than an automated test suite.

No third-party sample PDF is bundled with the repository. Use your own readable, text-based PDF for testing.

## Common Validation Areas

Both implementations have been manually tested for:

- PDF selection and validation
- readable-page extraction
- overlapping text chunk creation
- vector embedding generation
- Chroma indexing
- semantic retrieval
- grounded answer generation
- source-page reporting
- second-PDF replacement behavior
- clear-chat behavior
- full application reset
- UI state management
- serialized indexing and answer generation
- user-friendly error handling

## IBM watsonx.ai Test Environment

Tested with:

- Python 3.11
- IBM watsonx.ai
- IBM Granite embedding and chat models
- LangChain
- Chroma
- Gradio
- pypdf

The IBM version requires valid values for:

```text
WATSONX_APIKEY
WATSONX_PROJECT_ID
WATSONX_URL
```

## IBM Application Launch

Run:

```bash
python app.py
```

Expected behavior:

- Gradio starts successfully
- the interface is available at the local URL shown in the terminal
- a PDF can be uploaded and indexed
- questions can be asked after successful indexing
- generated answers include source pages

## IBM Pipeline Test

The current IBM command-line test in `rag_pipeline.py` expects a local PDF at:

```text
sample_docs/test.pdf
```

The repository intentionally does not include that PDF. To run the IBM CLI test, place your own readable PDF at that local path. Files matching `sample_docs/*.pdf` are ignored by Git.

Then run:

```bash
python rag_pipeline.py
```

Expected behavior:

- the PDF loads successfully
- readable pages are extracted
- chunks are created
- IBM watsonx.ai embeddings are generated
- Chroma stores the chunks
- semantic retrieval returns passages
- the Granite model generates a grounded answer
- source pages are printed

## Local Llama Test Environment

Tested with:

- Python 3.11
- Ollama
- Meta Llama `llama3.2`
- `embeddinggemma`
- LangChain
- Chroma
- Gradio
- pypdf

Ollama must be running and the configured models must be installed.

## Local Llama Application Launch

Run:

```bash
python llama/app_llama.py
```

Expected behavior:

- Gradio starts successfully
- a PDF can be uploaded and indexed locally
- questions can be asked after successful indexing
- answers are generated through Ollama
- source pages are displayed

## Local Llama Command-Line Test

Supply your own readable PDF:

```bash
python llama/rag_pipeline_llama.py /path/to/document.pdf "What is this document about?"
```

Expected behavior:

- the PDF loads successfully
- readable pages are extracted
- chunks are created
- `embeddinggemma` generates embeddings through Ollama
- Chroma indexes the chunks
- Llama generates an answer grounded in retrieved context
- source pages are printed

## Broad Document Question Validation

The local Llama implementation includes query-aware retrieval for broad document questions.

Test examples:

```text
What is the purpose of this PDF?
Summarize this article.
What is this document about?
```

Expected behavior:

- the broad-query path is detected
- retrieval expands beyond the normal specific-question setting
- introductory context from the beginning of the PDF is prioritized
- the answer directly addresses the high-level question when the retrieved context supports it
- source pages include relevant introductory pages when available

## Specific Question Validation

After indexing a PDF, ask a focused question about a clearly stated detail.

Expected behavior:

- normal similarity retrieval is used
- the answer stays within the retrieved document context
- source pages correspond to retrieved passages

## Blank Question

Steps:

1. Index a PDF.
2. Leave the question field blank.
3. Attempt to submit.

Expected behavior:

- no invalid response is added to the chat
- no unnecessary model request is made

## Ask Before Indexing

Steps:

1. Launch either application.
2. Do not index a PDF.
3. Attempt to ask a question.

Expected behavior:

- the Ask control remains unavailable until a document is indexed

## Invalid File Type

Attempt to select a non-PDF file.

Expected behavior:

- unsupported file types are rejected
- only PDF documents are accepted

## PDF With No Readable Text

Use a PDF containing no extractable text.

Expected behavior:

- indexing fails gracefully
- the application reports that no readable text was found

## Second PDF Replaces First PDF

Steps:

1. Index a first PDF.
2. Ask a question and verify the result.
3. Index a second PDF.
4. Ask a question related only to the first PDF.

Expected behavior:

- previous Chroma contents are reset
- the second PDF becomes the active document
- retrieval no longer uses chunks from the first PDF

## Race Condition Protection

While indexing or generating an answer, attempt to use conflicting controls.

Expected behavior:

- conflicting controls are disabled during the active operation
- indexing and answer generation remain serialized
- the application does not execute overlapping RAG operations

## IBM Token Quota Handling

If IBM watsonx.ai returns a token quota error such as:

```text
token_quota_reached
```

Expected behavior:

- the raw IBM traceback is not exposed in the Gradio interface
- the UI displays a specific IBM watsonx.ai token quota warning
- technical details remain available in terminal logs

## Ollama Availability Handling

Stop Ollama or use a configuration where a required local model is unavailable.

Expected behavior:

- the Gradio interface displays a user-friendly Ollama availability message
- internal exception details remain in terminal logs for troubleshooting

## Clear Chat

Steps:

1. Index a PDF.
2. Ask one or more questions.
3. Click **Clear Chat**.

Expected behavior:

- chat history is cleared
- the indexed PDF remains active
- additional questions can be asked without re-indexing

## Reset Application

Steps:

1. Index a PDF.
2. Ask a question.
3. Click **Reset**.

Expected behavior:

- chat history is cleared
- active retriever state is removed
- uploaded PDF selection is cleared
- Ask becomes disabled
- the interface returns to its initial state

## Known Testing Limitations

The project does not currently include automated unit or integration tests.

Testing has primarily focused on manual end-to-end validation of ingestion, retrieval, generation, source reporting, UI state, concurrency protection, and error handling.
