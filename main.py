# main.py

import argparse

from src.rag.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG System - question answering over your own PDFs, fully on-premise"
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "eval"],
        required=True,
        help="'ingest' indexes the PDFs in the raw data directory; 'query' asks questions about them; "
        "'eval' measures retrieval quality (add --answers to also judge generated answers).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question (query mode). If omitted, starts an interactive session.",
    )
    parser.add_argument("--answers", action="store_true", help="eval mode: also evaluate generated answers.")
    args = parser.parse_args()

    if args.mode == "eval":
        from src.rag.evaluate import evaluate_answers, evaluate_retrieval

        evaluate_retrieval()
        if args.answers:
            print("\n=== ANSWERS ===")
            evaluate_answers()
        return

    if args.mode == "ingest":
        from src.rag.index import create_index
        from src.rag.ingest import load_and_split_documents

        config = load_config()
        params = config["parameters"]
        print("\n=== INGESTION ===")
        chunks = load_and_split_documents(
            raw_data_path=config["paths"]["raw_data"],
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            min_letters=params.get("min_letters", 120),
        )
        if chunks:
            create_index(chunks, config)
        return

    from src.rag.index import IndexMismatchError
    from src.rag.llm import LLMError
    from src.rag.pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline()
    except IndexMismatchError as e:
        print(f"[Error] {e}")
        return

    def answer(question: str):
        try:
            result = pipeline.ask(question)
        except LLMError as e:
            print(f"[Error] {e}")
            return
        print(f"\n{result.answer}")
        if result.sources:
            print(f"\nSources: {result.format_sources()}")
        elif result.unverified:
            print("\n[Note] The model cited no fragment, so this answer could not be traced to a source.")

    if args.question:
        answer(args.question)
        return

    print("\nInteractive mode. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        answer(question)


if __name__ == "__main__":
    main()
