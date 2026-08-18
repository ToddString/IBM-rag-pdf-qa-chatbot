from pathlib import Path

import gradio as gr

from rag_pipeline_llama import (
    OLLAMA_ERROR_MESSAGE,
    format_sources,
    index_pdf,
    rag_answer,
)


APP_TITLE = "Local Llama Retrieval-Augmented Generation (RAG) PDF Question Answering Chatbot"
RAG_CONCURRENCY_ID = "llama_rag_operations"

INITIAL_STATUS = (
    "### No PDF indexed\n\n"
    "Upload a PDF document and click **Index PDF** to begin."
)

OLLAMA_STATUS = (
    "### ⚠️ Ollama unavailable\n\n"
    "The local Ollama service or one of the required models could not be reached.\n\n"
    "Make sure Ollama is running and that `llama3.2` and `embeddinggemma` are installed."
)


def is_ollama_error(error):
    error_text = str(error).lower()
    return (
        OLLAMA_ERROR_MESSAGE.lower() in error_text
        or "ollama" in error_text
        or "connection refused" in error_text
        or "failed to connect" in error_text
    )


def begin_indexing():
    return (
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.File(interactive=False),
        gr.Textbox(interactive=False),
        "### Indexing PDF...\n\nPlease wait until indexing is complete.",
    )


def process_pdf(pdf_path):
    if not pdf_path:
        return (
            None,
            "### No PDF selected\n\nPlease choose a PDF before clicking **Index PDF**.",
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(value="", interactive=False),
        )

    path = Path(pdf_path)

    if path.suffix.lower() != ".pdf":
        return (
            None,
            "### Invalid file type\n\nOnly PDF documents are supported.",
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(value="", interactive=False),
        )

    try:
        document_retriever, page_count, chunk_count = index_pdf(
            str(path),
            batch_size=8,
            k=3,
        )

        status = (
            "### ✅ PDF indexed successfully\n\n"
            f"**File:** `{path.name}`\n\n"
            f"**Readable pages:** {page_count}\n\n"
            f"**Chunks indexed:** {chunk_count}\n\n"
            "You can now ask questions about the document using local Llama."
        )

        return (
            document_retriever,
            status,
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(value="", interactive=True),
        )

    except Exception as error:
        print(f"PDF indexing error for {path.name}: {error}")

        status = OLLAMA_STATUS if is_ollama_error(error) else (
            "### ❌ Unable to index PDF\n\n"
            "The document could not be processed. Check the terminal for technical details."
        )

        return (
            None,
            status,
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(value="", interactive=False),
        )


def begin_question():
    return (
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.File(interactive=False),
        gr.Textbox(interactive=False),
    )


def finish_question():
    return (
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.File(interactive=True),
        gr.Textbox(interactive=True),
    )


def ask_question(question, document_retriever, history):
    history = history or []

    if document_retriever is None:
        response = "Please upload and index a PDF before asking a question."
        return (
            history
            + [
                {"role": "user", "content": question or ""},
                {"role": "assistant", "content": response},
            ],
            question or "",
        )

    if not question or not question.strip():
        return history, ""

    question = question.strip()

    try:
        answer, source_documents = rag_answer(document_retriever, question)
        sources = format_sources(source_documents)

        if sources:
            source_lines = "\n".join(f"- {source}" for source in sources)
            response = f"{answer}\n\n### Sources\n{source_lines}"
        else:
            response = answer

    except Exception as error:
        print(f"Question answering error: {error}")

        if is_ollama_error(error):
            response = (
                "⚠️ **Ollama unavailable**\n\n"
                "Make sure the local Ollama service is running and the required models are installed."
            )
        else:
            response = (
                "❌ **Unable to generate an answer.**\n\n"
                "An error occurred while processing the question. Check the terminal for technical details."
            )

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    return updated_history, ""


def clear_chat():
    return []


def reset_application():
    return (
        None,
        INITIAL_STATUS,
        [],
        None,
        gr.Button(interactive=True),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Textbox(value="", interactive=False),
    )


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        f"# {APP_TITLE}\n\n"
        "Upload a PDF, index it locally, and ask grounded questions with Meta Llama through Ollama.\n\n"
        "**Technology stack:** Ollama, Meta Llama, LangChain, Chroma, pypdf, and Gradio."
    )

    document_retriever_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Document")
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            index_button = gr.Button("Index PDF", variant="primary")
            status_output = gr.Markdown(INITIAL_STATUS)

        with gr.Column(scale=2):
            gr.Markdown("## Ask Questions")
            chatbot = gr.Chatbot(label="Document Q&A", height=500)
            question_input = gr.Textbox(
                label="Question",
                placeholder="Index a PDF before asking a question...",
                lines=2,
                interactive=False,
            )

            with gr.Row():
                ask_button = gr.Button(
                    "Ask",
                    variant="primary",
                    interactive=False,
                )
                clear_button = gr.Button("Clear Chat")
                reset_button = gr.Button("Reset")

    indexing_event = index_button.click(
        fn=begin_indexing,
        inputs=None,
        outputs=[
            index_button,
            ask_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
            status_output,
        ],
        queue=False,
    )

    indexing_event.then(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[
            document_retriever_state,
            status_output,
            chatbot,
            index_button,
            ask_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
        ],
        concurrency_id=RAG_CONCURRENCY_ID,
        concurrency_limit=1,
        show_progress="full",
    )

    ask_event = ask_button.click(
        fn=begin_question,
        inputs=None,
        outputs=[
            ask_button,
            index_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
        ],
        queue=False,
    )

    ask_event = ask_event.then(
        fn=ask_question,
        inputs=[question_input, document_retriever_state, chatbot],
        outputs=[chatbot, question_input],
        concurrency_id=RAG_CONCURRENCY_ID,
        concurrency_limit=1,
        show_progress="minimal",
    )

    ask_event.then(
        fn=finish_question,
        inputs=None,
        outputs=[
            ask_button,
            index_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
        ],
        queue=False,
    )

    submit_event = question_input.submit(
        fn=begin_question,
        inputs=None,
        outputs=[
            ask_button,
            index_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
        ],
        queue=False,
    )

    submit_event = submit_event.then(
        fn=ask_question,
        inputs=[question_input, document_retriever_state, chatbot],
        outputs=[chatbot, question_input],
        concurrency_id=RAG_CONCURRENCY_ID,
        concurrency_limit=1,
        show_progress="minimal",
    )

    submit_event.then(
        fn=finish_question,
        inputs=None,
        outputs=[
            ask_button,
            index_button,
            reset_button,
            clear_button,
            pdf_input,
            question_input,
        ],
        queue=False,
    )

    clear_button.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot],
        queue=False,
    )

    reset_button.click(
        fn=reset_application,
        inputs=None,
        outputs=[
            document_retriever_state,
            status_output,
            chatbot,
            pdf_input,
            index_button,
            ask_button,
            reset_button,
            clear_button,
            question_input,
        ],
        queue=False,
    )


if __name__ == "__main__":
    print("=" * 60)
    print(APP_TITLE)
    print("=" * 60)
    print("Starting Gradio application...")

    demo.queue(default_concurrency_limit=1)
    demo.launch()
