# src/rag/pipeline.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from src.rag.config import load_config
from src.rag.index import load_index
from src.rag.llm import answer_schema, create_llm
from src.rag.retrieve import Hit, Retriever

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class RAGAnswer:
    answer: str
    sources: List[Hit] = field(default_factory=list)  # only the fragments the model cited
    answerable: bool = True
    unverified: bool = False  # the model answered but cited no valid fragment

    def format_sources(self) -> str:
        seen = []
        for hit in self.sources:
            ref = f"{hit.chunk.metadata.get('source', '?')} (page {hit.chunk.metadata.get('page', '?')})"
            if ref not in seen:
                seen.append(ref)
        return ", ".join(seen)


def load_prompt(name: str) -> Tuple[str, str]:
    """Reads src/rag/prompts/<name>.md and returns its (system, user) templates."""
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    if "---system---" not in text or "---user---" not in text:
        raise ValueError(f"Prompt {name!r} must contain ---system--- and ---user--- sections")
    _, rest = text.split("---system---", 1)
    system, user = rest.split("---user---", 1)
    return system.strip(), user.strip()


def build_context(hits: Sequence[Hit]) -> str:
    """Numbers each fragment and tells the model where it comes from, so it can cite it."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        meta = hit.chunk.metadata
        blocks.append(f"[{n}] {meta.get('source', '?')}, p.{meta.get('page', '?')}\n{hit.chunk.page_content}")
    return "\n\n".join(blocks)


def valid_citations(citations: Sequence[int], n_fragments: int) -> List[int]:
    """Keeps only citations that point to a fragment actually given to the model (1-based, unique, ordered)."""
    seen: List[int] = []
    for c in citations:
        if isinstance(c, int) and 1 <= c <= n_fragments and c not in seen:
            seen.append(c)
    return seen


class RAGPipeline:
    """
    Loads the index, the reranker and the Ollama client once, so many questions can be
    answered without paying the start-up cost each time. `retriever` and `llm` can be
    injected, which is how the tests run without models.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        retriever: Optional[Retriever] = None,
        llm=None,
    ):
        self.config = load_config(config_path)
        if retriever is None:
            print("[Pipeline] Loading index and embedding model...")
            retriever = Retriever(self.config, load_index(self.config))
        self.retriever = retriever
        self.llm = llm or create_llm(self.config)

        prompts_cfg = self.config["prompts"]
        self.system_prompt, self.user_template = load_prompt(prompts_cfg["template"])
        self.fallback = prompts_cfg["fallback"]

    def ask(self, question: str) -> RAGAnswer:
        hits = self.retriever.retrieve(question)
        if not hits:
            return RAGAnswer(self.fallback, answerable=False)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(context=build_context(hits), question=question)},
        ]
        reply = self.llm.chat_json(messages, answer_schema(len(hits)))

        answer = str(reply.get("answer", "")).strip()
        if not reply.get("answerable") or not answer:
            return RAGAnswer(self.fallback, answerable=False)

        cited = valid_citations(reply.get("citations", []), len(hits))
        return RAGAnswer(answer, [hits[n - 1] for n in cited], unverified=not cited)
