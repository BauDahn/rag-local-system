# src/rag/evaluate.py
"""
Evaluation of the RAG system.

Retrieval (default, no LLM): for each answerable question checks whether any of the top-k
chunks contains an expected keyword (hit rate) and at which rank the first one appears
(MRR). It runs the same questions through dense, hybrid and hybrid+rerank retrieval so the
effect of each stage is visible.

With --answers it also runs the full pipeline: questions marked `answerable: false` must
be refused, answerable ones must contain one of their `expected` words (answer accuracy,
independent of any LLM), and an LLM judge additionally checks faithfulness to the
fragments the model cited. The judge is the same model that answers, so treat that number
as a weak signal.

Usage: python -m src.rag.evaluate [--answers] [--config configs/x.yaml] [--eval-set path/to/eval_set.yaml]
"""

import argparse
import re
from typing import List, Optional

import yaml

from src.rag.config import ROOT, load_config
from src.rag.index import load_index
from src.rag.llm import LLMError
from src.rag.retrieve import MODES, Retriever

DEFAULT_EVAL_SET = ROOT / "data" / "eval_set.yaml"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}


def contains_keyword(text: str, keywords: List[str]) -> bool:
    """Whole-word, case-insensitive match, so 'boa' does not match 'boat'. Whitespace (PDF line breaks) is collapsed."""
    text = " ".join(text.split())
    return any(re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE) for kw in keywords)


def first_relevant_rank(texts: List[str], keywords: List[str]) -> Optional[int]:
    return next((i for i, t in enumerate(texts, 1) if contains_keyword(t, keywords)), None)


def resolve_eval_path(config: dict, eval_path=None):
    """Explicit argument > `paths.eval_set` from the config > data/eval_set.yaml."""
    return eval_path or config["paths"].get("eval_set") or DEFAULT_EVAL_SET


def load_cases(eval_path) -> List[dict]:
    with open(eval_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_retrieval(eval_path=None, config_path: Optional[str] = None) -> dict:
    config = load_config(config_path)
    eval_path = resolve_eval_path(config, eval_path)
    retriever = Retriever(config, load_index(config))
    k = retriever.top_k
    cases = [c for c in load_cases(eval_path) if c.get("answerable", True)]

    results = {}
    for mode in MODES:
        hits, reciprocal_ranks = 0, []
        for case in cases:
            texts = [h.chunk.page_content for h in retriever.retrieve(case["question"], mode)]
            rank = first_relevant_rank(texts, case["keywords"])
            hits += rank is not None
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            print(f"[{mode:13}] [{'HIT ' if rank else 'MISS'}] {case['question']}")
        results[mode] = {f"hit_rate@{k}": hits / len(cases), "mrr": sum(reciprocal_ranks) / len(cases)}

    print(f"\n{'mode':14} {'hit_rate@' + str(k):>12} {'mrr':>8}   ({len(cases)} questions)")
    for mode, m in results.items():
        print(f"{mode:14} {m[f'hit_rate@{k}']:>12.2f} {m['mrr']:>8.2f}")
    return results


def evaluate_answers(eval_path=None, config_path: Optional[str] = None) -> dict:
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(config_path)
    cases = load_cases(resolve_eval_path(pipeline.config, eval_path))

    refused_ok, refused_total = 0, 0
    correct, answerable_total = 0, 0
    faithful, faithful_total = 0, 0
    llm_errors, answerable_errors = 0, 0
    for case in cases:
        try:
            result = pipeline.ask(case["question"])
        except LLMError as e:  # e.g. truncated or invalid JSON: count it as a failure, keep going
            llm_errors += 1
            print(f"[ERROR] {case['question']}\n        {e}")
            if not case.get("answerable", True):
                refused_total += 1
            else:
                answerable_total += 1
                answerable_errors += 1
            continue
        if not case.get("answerable", True):
            refused_total += 1
            refused_ok += not result.answerable
            print(f"[{'OK  ' if not result.answerable else 'FAIL'}] should refuse: {case['question']}")
            continue

        answerable_total += 1
        if not result.answerable:
            print(f"[REFUSED] {case['question']}")
            continue

        is_correct = contains_keyword(result.answer, case["expected"]) if case.get("expected") else None
        correct += bool(is_correct)

        evidence = "\n\n".join(h.chunk.page_content for h in result.sources) or "(no fragment was cited)"
        verdict = pipeline.llm.chat_json(
            [
                {"role": "system", "content": "You check whether an answer is fully supported by the given evidence. "
                                              "Reply with JSON: supported (true only if every claim is in the evidence) and reason."},
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {case['question']}\nAnswer: {result.answer}"},
            ],
            JUDGE_SCHEMA,
        )
        faithful_total += 1
        faithful += bool(verdict.get("supported"))
        tag = {True: "OK  ", False: "WRONG", None: "?   "}[is_correct]
        print(f"[{tag}] {case['question']}\n        {result.answer}")

    metrics = {
        "refusal_accuracy": refused_ok / refused_total if refused_total else None,
        "answer_accuracy": correct / answerable_total if answerable_total else None,
        "refused_answerable": (answerable_total - faithful_total - answerable_errors) / answerable_total if answerable_total else None,
        "faithfulness_self_judged": faithful / faithful_total if faithful_total else None,
    }
    print(f"\n{refused_total} unanswerable, {answerable_total} answerable questions, {llm_errors} LLM errors")
    print(", ".join(f"{k}: {v:.2f}" for k, v in metrics.items() if v is not None))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="Also evaluate generated answers (needs Ollama).")
    parser.add_argument("--eval-set", default=None, help="Default: paths.eval_set from the config, else data/eval_set.yaml.")
    parser.add_argument("--config", default=None, help="Config profile (default: config.yaml).")
    args = parser.parse_args()

    evaluate_retrieval(args.eval_set, args.config)
    if args.answers:
        print("\n=== ANSWERS ===")
        evaluate_answers(args.eval_set, args.config)
