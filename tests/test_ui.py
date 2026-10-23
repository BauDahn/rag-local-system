from langchain_core.documents import Document

from src.rag.index import IndexMismatchError
from src.rag.llm import LLMError
from src.rag.pipeline import RAGAnswer
from src.rag.retrieve import Hit
from src.rag.ui import error_message, format_response, format_sources


def hit(text, source="book.pdf", page=3):
    return Hit(Document(page_content=text, metadata={"source": source, "page": page}), 1.0)


def test_format_sources_rows_and_truncation():
    rows = format_sources(RAGAnswer("a", [hit("x  y\nz"), hit("w" * 500)]), max_chars=20)
    assert rows[0] == ["book.pdf", "3", "x y z"]
    assert rows[1][2] == "w" * 20 + "..."


def test_format_response_unverified_and_refusal():
    answer, rows, note = format_response(RAGAnswer("hi", unverified=True))
    assert answer == "hi" and rows == [] and "could not be traced" in note
    assert format_response(RAGAnswer("I do not know", answerable=False))[1:] == ([], "")


def test_error_messages():
    assert "Ollama" in error_message(LLMError("down"))
    assert "index" in error_message(IndexMismatchError("bad"))
