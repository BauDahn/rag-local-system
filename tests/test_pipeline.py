from langchain_core.documents import Document

from src.rag.pipeline import RAGPipeline, build_context, load_prompt, valid_citations
from src.rag.retrieve import Hit


def make_hits(n=3):
    return [Hit(Document(page_content=f"text {i}", metadata={"source": "book.pdf", "page": i}), 1.0) for i in range(1, n + 1)]


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, mode="hybrid_rerank"):
        return self.hits


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None

    def chat_json(self, messages, schema=None):
        self.messages = messages
        return self.reply


def pipeline(hits, reply):
    return RAGPipeline(retriever=FakeRetriever(hits), llm=FakeLLM(reply))


def test_load_prompt_splits_sections():
    system, user = load_prompt("qa")
    assert "citations" in system
    assert "{context}" in user and "{question}" in user


def test_build_context_labels_each_fragment():
    context = build_context(make_hits(2))
    assert "[1] book.pdf, p.1\ntext 1" in context
    assert "[2] book.pdf, p.2\ntext 2" in context


def test_valid_citations_drops_out_of_range_duplicates_and_non_ints():
    assert valid_citations([2, 2, 9, 0, "1", 1], 3) == [2, 1]


def test_answer_with_valid_citations_returns_only_cited_sources():
    p = pipeline(make_hits(3), {"answerable": True, "answer": "A fox.", "citations": [3]})
    result = p.ask("Who?")
    assert result.answer == "A fox."
    assert [h.chunk.metadata["page"] for h in result.sources] == [3]
    assert not result.unverified


def test_answer_without_valid_citations_is_flagged_unverified():
    p = pipeline(make_hits(2), {"answerable": True, "answer": "Something.", "citations": [7]})
    result = p.ask("Who?")
    assert result.answer == "Something." and result.sources == [] and result.unverified


def test_not_answerable_returns_fallback_without_sources():
    p = pipeline(make_hits(2), {"answerable": False, "answer": "", "citations": [1]})
    result = p.ask("Capital of France?")
    assert result.answer == p.fallback and not result.answerable and result.sources == []


def test_no_hits_skips_the_llm():
    p = pipeline([], {"answerable": True, "answer": "x", "citations": [1]})
    result = p.ask("Anything?")
    assert result.answer == p.fallback and p.llm.messages is None


def test_prompt_contains_question_and_context():
    p = pipeline(make_hits(1), {"answerable": True, "answer": "ok", "citations": [1]})
    p.ask("What is it?")
    user = p.llm.messages[1]["content"]
    assert "What is it?" in user and "[1] book.pdf, p.1" in user
