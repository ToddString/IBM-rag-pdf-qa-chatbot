from pathlib import Path

import gradio as gr

from rag_pipeline import (
    QUOTA_ERROR_MESSAGE,
    format_sources,
    index_pdf,
    rag_answer,
)


# =========================================================
# Application configuration
# =========================================================

APP_TITLE = (
    "Retrieval-Augmented Generation (RAG) "
    "PDF Question Answering Chatbot"
)

RAG_CONCURRENCY_ID = "rag_operations"

INITIAL_STATUS = (
    "### No PDF indexed\n\n"
    "Upload a PDF document and click **Index PDF** "
    "to begin."
)

QUOTA_STATUS = (
    "### ⚠️ IBM watsonx.ai Token Quota Reached\n\n"
    "IBM watsonx.ai has reported that the available "
    "token quota has been reached.\n\n"
    "The request cannot be completed until additional "
    "token capacity is available."
)


# =========================================================
# Error helpers
# =========================================================

def is_quota_error(error):
    """
    Determine whether an exception represents an IBM
    watsonx.ai token quota error.
    """
    error_text = str(error)

    return (
        QUOTA_ERROR_MESSAGE in error_text
        or "token_quota_reached" in error_text
        or "Token consumption quota has been reached"
        in error_text
    )


# =========================================================
# PDF indexing controls
# =========================================================

def begin_indexing():
    """
    Disable interactive controls while a PDF is being
    indexed.
    """
    return (
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.File(interactive=False),
        gr.Textbox(interactive=False),
        (
            "### Indexing PDF...\n\n"
            "Please wait until indexing is complete."
        ),
    )


def process_pdf(pdf_path):
    """
    Load and index the selected PDF.

    Returns the retriever, interface status, cleared chat
    history, and updated component states.
    """
    if not pdf_path:
        return (
            None,
            (
                "### No PDF selected\n\n"
                "Please choose a PDF file before "
                "clicking **Index PDF**."
            ),
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(
                value="",
                interactive=False,
            ),
        )

    path = Path(pdf_path)

    if path.suffix.lower() != ".pdf":
        return (
            None,
            (
                "### Invalid file type\n\n"
                "Only PDF documents are supported."
            ),
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(
                value="",
                interactive=False,
            ),
        )

    print()
    print("=" * 60)
    print(f"Indexing PDF: {path.name}")
    print("=" * 60)

    try:
        (
            document_retriever,
            page_count,
            chunk_count,
        ) = index_pdf(
            str(path),
            batch_size=8,
            k=3,
        )

        status = (
            "### ✅ PDF indexed successfully\n\n"
            f"**File:** `{path.name}`\n\n"
            f"**Readable pages:** {page_count}\n\n"
            f"**Chunks indexed:** {chunk_count}\n\n"
            "You can now ask questions about the "
            "document."
        )

        print(
            f"Successfully indexed {path.name}"
        )
        print(
            f"Readable pages: {page_count}"
        )
        print(
            f"Chunks indexed: {chunk_count}"
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
            gr.Textbox(
                value="",
                interactive=True,
            ),
        )

    except Exception as error:
        print(
            f"PDF indexing error for "
            f"{path.name}: {error}"
        )

        if is_quota_error(error):
            return (
                None,
                QUOTA_STATUS,
                [],
                gr.Button(interactive=True),
                gr.Button(interactive=False),
                gr.Button(interactive=True),
                gr.Button(interactive=True),
                gr.File(interactive=True),
                gr.Textbox(
                    value="",
                    interactive=False,
                ),
            )

        return (
            None,
            (
                "### ❌ Unable to index PDF\n\n"
                "The document could not be processed.\n\n"
                "Check the terminal for technical "
                "details."
            ),
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.File(interactive=True),
            gr.Textbox(
                value="",
                interactive=False,
            ),
        )


# =========================================================
# Question controls
# =========================================================

def begin_question():
    """
    Disable controls while retrieval and answer generation
    are running.
    """
    return (
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.File(interactive=False),
        gr.Textbox(interactive=False),
    )


def finish_question():
    """
    Re-enable controls after question answering finishes.
    """
    return (
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.File(interactive=True),
        gr.Textbox(interactive=True),
    )


# =========================================================
# Question answering
# =========================================================

def ask_question(
    question,
    document_retriever,
    history,
):
    """
    Ask a question about the currently indexed PDF and
    append the grounded response to the chat history.
    """
    history = history or []

    if document_retriever is None:
        response = (
            "Please upload and index a PDF before "
            "asking a question."
        )

        updated_history = history + [
            {
                "role": "user",
                "content": question or "",
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return (
            updated_history,
            question or "",
        )

    if not question or not question.strip():
        return (
            history,
            "",
        )

    question = question.strip()

    print()
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    try:
        (
            answer,
            source_documents,
        ) = rag_answer(
            document_retriever,
            question,
        )

        sources = format_sources(
            source_documents
        )

        if sources:
            source_lines = "\n".join(
                f"- {source}"
                for source in sources
            )

            response = (
                f"{answer}\n\n"
                f"### Sources\n"
                f"{source_lines}"
            )

        else:
            response = answer

        print()
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        updated_history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return (
            updated_history,
            "",
        )

    except Exception as error:
        print(
            f"Question answering error: {error}"
        )

        if is_quota_error(error):
            response = (
                "⚠️ **IBM watsonx.ai Token Quota "
                "Reached**\n\n"
                "IBM watsonx.ai has reported that the "
                "available token quota has been "
                "reached.\n\n"
                "The question cannot be answered until "
                "additional token capacity is available."
            )

        else:
            response = (
                "❌ **Unable to generate an answer.**\n\n"
                "An error occurred while processing the "
                "question. Check the terminal for "
                "technical details."
            )

        updated_history = history + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        return (
            updated_history,
            question,
        )


# =========================================================
# Chat and application reset
# =========================================================

def clear_chat():
    """
    Clear the current conversation without removing the
    indexed PDF.
    """
    return []


def reset_application():
    """
    Reset the interface and remove the active retriever
    from application state.
    """
    return (
        None,
        INITIAL_STATUS,
        [],
        None,
        gr.Button(interactive=True),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Textbox(
            value="",
            interactive=False,
        ),
    )


# =========================================================
# Gradio interface
# =========================================================

with gr.Blocks(
    title=APP_TITLE,
) as demo:

    gr.Markdown(
        f"# {APP_TITLE}\n\n"
        "Upload a PDF document, index its contents, "
        "and ask questions using retrieval-augmented "
        "generation.\n\n"
        "**Technology stack:** IBM watsonx.ai, "
        "LangChain, Chroma, pypdf, and Gradio."
    )

    document_retriever_state = gr.State(
        value=None
    )

    with gr.Row():

        # -------------------------------------------------
        # Document panel
        # -------------------------------------------------

        with gr.Column(
            scale=1
        ):
            gr.Markdown(
                "## Document"
            )

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            index_button = gr.Button(
                "Index PDF",
                variant="primary",
            )

            status_output = gr.Markdown(
                INITIAL_STATUS
            )

        # -------------------------------------------------
        # Chat panel
        # -------------------------------------------------

        with gr.Column(
            scale=2
        ):
            gr.Markdown(
                "## Ask Questions"
            )

            chatbot = gr.Chatbot(
                label="Document Q&A",
                height=500,
            )

            question_input = gr.Textbox(
                label="Question",
                placeholder=(
                    "Index a PDF before asking "
                    "a question..."
                ),
                lines=2,
                interactive=False,
            )

            with gr.Row():
                ask_button = gr.Button(
                    "Ask",
                    variant="primary",
                    interactive=False,
                )

                clear_button = gr.Button(
                    "Clear Chat"
                )

                reset_button = gr.Button(
                    "Reset"
                )


    # =====================================================
    # Index PDF event
    # =====================================================

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
        inputs=[
            pdf_input,
        ],
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


    # =====================================================
    # Ask button event
    # =====================================================

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
        inputs=[
            question_input,
            document_retriever_state,
            chatbot,
        ],
        outputs=[
            chatbot,
            question_input,
        ],
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


    # =====================================================
    # Enter key event
    # =====================================================

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
        inputs=[
            question_input,
            document_retriever_state,
            chatbot,
        ],
        outputs=[
            chatbot,
            question_input,
        ],
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


    # =====================================================
    # Clear chat event
    # =====================================================

    clear_button.click(
        fn=clear_chat,
        inputs=None,
        outputs=[
            chatbot,
        ],
        queue=False,
    )


    # =====================================================
    # Reset application event
    # =====================================================

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


# =========================================================
# Application entry point
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print(APP_TITLE)
    print("=" * 60)
    print("Starting Gradio application...")

    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch()