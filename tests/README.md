Testing Guide

This directory documents the manual validation performed for the RAG PDF Question Answering Chatbot.

The project currently uses manual end-to-end testing rather than an automated test suite.

Test Environment

Tested with:

Python 3.11

IBM watsonx.ai

IBM Granite embedding and chat models

LangChain

Chroma

Gradio

pypdf

Test 1: Pipeline End-to-End Validation

Run:

python rag_pipeline.py

Expected behavior:

sample PDF loads successfully

readable pages are extracted

text chunks are created

IBM watsonx.ai embeddings are generated

Chroma stores the document chunks

semantic retrieval returns relevant passages

the Granite chat model generates a grounded answer

source pages are displayed

Example test question:

What is Low-Rank Adaptation?

Successful test results included:

Readable pages: 11
Chunks indexed: 38

The generated answer was grounded in retrieved passages from the source PDF.

Test 2: Application Launch

Run:

python app.py

Expected behavior:

Gradio starts successfully

the local interface is available at:

http://127.0.0.1:7860

Test 3: PDF Upload and Indexing

Steps:

Open the Gradio interface.

Upload a valid PDF.

Click Index PDF.

Wait for indexing to complete.

Expected behavior:

indexing status is displayed

controls are temporarily disabled during indexing

readable page count is displayed

chunk count is displayed

Ask becomes available after successful indexing

Test 4: Question Answering

Steps:

Index a valid PDF.

Enter a question about the document.

Click Ask or press Enter.

Expected behavior:

relevant document chunks are retrieved

the IBM Granite chat model generates an answer

the answer is based only on retrieved document context

source pages are displayed below the answer

Test 5: Blank Question

Steps:

Index a valid PDF.

Leave the question field blank.

Attempt to submit.

Expected behavior:

no API request is made

no invalid response is added to the chat

Test 6: Ask Before Indexing

Steps:

Launch the application.

Do not index a PDF.

Attempt to ask a question.

Expected behavior:

the Ask control remains disabled until a document is indexed

Test 7: Invalid File Type

Steps:

Attempt to select a non-PDF file.

Expected behavior:

the application rejects unsupported file types

only PDF documents are accepted

Test 8: PDF With No Readable Text

Use a PDF containing no extractable text.

Expected behavior:

indexing fails gracefully

the application reports that no readable text was found

Test 9: Second PDF Replaces First PDF

Steps:

Index the first PDF.

Ask a question and verify the result.

Index a second PDF.

Ask a question related only to the first PDF.

Expected behavior:

the previous Chroma contents are reset

the second PDF becomes the active document

retrieval no longer uses chunks from the first PDF

Test 10: Race Condition Protection

Steps:

Upload a larger PDF.

Click Index PDF.

While indexing is running, attempt to use other controls.

Expected behavior:

Ask is disabled during indexing

Reset is disabled during indexing

Clear Chat is disabled during indexing

PDF selection is disabled during indexing

the question field is disabled during indexing

conflicting RAG operations do not execute at the same time

Test 11: Answer Generation Concurrency

Steps:

Index a PDF.

Submit a question.

Attempt to trigger another document or chat operation while the answer is being generated.

Expected behavior:

conflicting controls remain disabled until answer generation finishes

retrieval and answer generation remain serialized through the Gradio event queue

Test 12: IBM watsonx.ai Token Quota Handling

If IBM watsonx.ai returns a token quota error such as:

token_quota_reached

Expected behavior:

the raw IBM traceback is not shown in the Gradio interface

the UI displays a specific IBM watsonx.ai token quota warning

technical details remain available in the terminal logs

Test 13: Clear Chat

Steps:

Index a PDF.

Ask one or more questions.

Click Clear Chat.

Expected behavior:

chat history is cleared

the indexed PDF remains available

additional questions can still be asked without re-indexing

Test 14: Reset Application

Steps:

Index a PDF.

Ask a question.

Click Reset.

Expected behavior:

chat history is cleared

active retriever state is removed

uploaded PDF selection is cleared

Ask becomes disabled

the interface returns to its initial state

Known Testing Limitations

The current project does not include automated unit or integration tests.

Testing has primarily focused on:

PDF ingestion

text extraction

chunk creation

IBM embedding requests

Chroma indexing

semantic retrieval

answer generation

source reporting

UI state management

concurrency protection

error handling

Automated tests may be added in a future version.