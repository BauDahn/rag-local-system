# src/rag/ui.py

from typing import List, Tuple

from src.rag.index import IndexMismatchError
from src.rag.llm import LLMError
from src.rag.pipeline import RAGAnswer

SNIPPET_CHARS = 300


def format_sources(result: RAGAnswer, max_chars: int = SNIPPET_CHARS) -> List[List[str]]:
    """Rows [file, page, fragment] for the cited sources (pure function, no models needed)."""
    rows = []
    for hit in result.sources:
        meta = hit.chunk.metadata
        text = " ".join(hit.chunk.page_content.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        rows.append([str(meta.get("source", "?")), str(meta.get("page", "?")), text])
    return rows


def format_response(result: RAGAnswer) -> Tuple[str, List[List[str]], str]:
    """Returns (answer, source rows, note) for display."""
    note = ""
    if result.unverified:
        note = "The model cited no fragment, so this answer could not be traced to a source."
    return result.answer, format_sources(result), note


def error_message(exc: Exception) -> str:
    if isinstance(exc, LLMError):
        return f"The language model (Ollama) is not responding: {exc}"
    if isinstance(exc, IndexMismatchError):
        return f"The index does not match the current configuration: {exc}"
    return f"Unexpected error: {exc}"


def build_app(pipeline=None, config_path=None):
    """Builds the Gradio app. The pipeline is loaded once (or injected) and reused for every question."""
    import gradio as gr

    state = {"pipeline": pipeline, "error": None}
    if state["pipeline"] is None:
        from src.rag.pipeline import RAGPipeline

        try:
            state["pipeline"] = RAGPipeline(config_path)
        except IndexMismatchError as e:
            state["error"] = error_message(e)

    def run(question: str):
        if state["error"]:
            return state["error"], [], ""
        question = (question or "").strip()
        if not question:
            return "Please type a question.", [], ""
        try:
            result = state["pipeline"].ask(question)
        except (LLMError, IndexMismatchError) as e:
            return error_message(e), [], ""
        return format_response(result)

    with gr.Blocks(title="Local RAG") as app:
        gr.Markdown("# Local RAG\nAsk questions about your documents. Everything runs on this machine.")
        question = gr.Textbox(label="Question", lines=2, placeholder="What does the prince look like?")
        button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=6, interactive=False)
        note = gr.Markdown()
        sources = gr.Dataframe(
            headers=["File", "Page", "Fragment"], datatype=["str", "str", "str"],
            label="Cited sources", interactive=False, wrap=True, column_widths=["14%", "8%", "78%"],
        )
        for trigger in (button.click, question.submit):
            trigger(run, inputs=question, outputs=[answer, sources, note])
    return app


def launch(port: int = 7860, config_path=None):
    build_app(config_path=config_path).launch(server_name="127.0.0.1", server_port=port)
