# src/rag/retrieve.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from langchain_core.documents import Document

from src.rag.index import Index, tokenize

MODES = ("dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Document
    score: float  # reranker score, RRF score or negated rank depending on the mode


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    """
    Merges several ranked lists of chunk ids. Each list contributes 1 / (k + rank) per id,
    so documents ranked well by several retrievers rise to the top without having to
    compare their (incompatible) raw scores.
    """
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, config: dict, index: Index):
        self.index = index
        params = config["parameters"]
        self.k_candidates = params.get("k_candidates", 20)
        self.top_k = params.get("top_k", 4)
        self.rrf_k = params.get("rrf_k", 60)
        self.min_rerank_score: Optional[float] = params.get("min_rerank_score")
        self._reranker_name = config["models"]["reranker"]
        self._reranker_device = config["models"].get("reranker_device", "cpu")
        self._reranker = None  # loaded on first use so dense/hybrid runs never pay for it

    def dense_ranking(self, query: str, n: int) -> List[int]:
        vector = self.index.embeddings.embed_query(self.index.query_prefix + query)
        docs = self.index.db.similarity_search_by_vector(vector, k=n)
        return [d.metadata["chunk_id"] for d in docs]

    def bm25_ranking(self, query: str, n: int) -> List[int]:
        scores = self.index.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:n]
        return [int(i) for i in order if scores[i] > 0]

    def _rerank(self, query: str, chunk_ids: List[int]) -> List[Hit]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._reranker_name, device=self._reranker_device)
        chunks = [self.index.chunks[i] for i in chunk_ids]
        scores = self._reranker.predict([(query, c.page_content) for c in chunks])
        hits = [Hit(c, float(s)) for c, s in zip(chunks, scores)]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def retrieve(self, query: str, mode: str = "hybrid_rerank") -> List[Hit]:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {MODES}")

        dense = self.dense_ranking(query, self.k_candidates)
        if mode == "dense":
            ids = dense[: self.top_k]
            return [Hit(self.index.chunks[i], -rank) for rank, i in enumerate(ids)]

        fused = reciprocal_rank_fusion([dense, self.bm25_ranking(query, self.k_candidates)], self.rrf_k)
        ranked = sorted(fused, key=fused.get, reverse=True)
        if mode == "hybrid":
            return [Hit(self.index.chunks[i], fused[i]) for i in ranked[: self.top_k]]

        hits = self._rerank(query, ranked)
        if self.min_rerank_score is not None:
            hits = [h for h in hits if h.score >= self.min_rerank_score]
        return hits[: self.top_k]
