# src/evaluate.py
"""
Retrieval evaluation: for each test question, checks whether the top-k retrieved
chunks contain at least one expected keyword (hit rate) and at which rank the first
relevant chunk appears (MRR). It needs no LLM, so it runs fast.

Usage: python -m src.evaluate [path/to/eval_set.yaml]
"""

import sys

import yaml

from src.embed import build_embeddings, load_config, load_vector_store

DEFAULT_EVAL_SET = "data/eval_set.yaml"


def evaluate_retrieval(eval_path: str = DEFAULT_EVAL_SET, config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    k = config["parameters"].get("top_k", 3)
    db = load_vector_store(config, build_embeddings(config))

    with open(eval_path, "r") as f:
        cases = yaml.safe_load(f)

    hits, reciprocal_ranks = 0, []
    for case in cases:
        scored = db.similarity_search_with_score(case["question"], k=k)
        docs = [d for d, _ in scored]
        print(f"      distances: {[round(dist, 3) for _, dist in scored]}")
        keywords = [kw.lower() for kw in case["keywords"]]
        rank = next(
            (i for i, d in enumerate(docs, 1) if any(kw in d.page_content.lower() for kw in keywords)),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"[{'HIT ' if rank else 'MISS'}] {case['question']}")

    n = len(cases)
    metrics = {f"hit_rate@{k}": hits / n, "mrr": sum(reciprocal_ranks) / n, "questions": n}
    print("\n" + ", ".join(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}"
                           for name, value in metrics.items()))
    return metrics


if __name__ == "__main__":
    evaluate_retrieval(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_SET)
