# main.py

import argparse

from src.embed import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG System - question answering over your own PDFs, fully on-premise"
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "query"],
        required=True,
        help="'ingest' indexes the PDFs in the raw data directory; 'query' asks questions about them.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question (query mode). If omitted, starts an interactive session.",
    )
    args = parser.parse_args()
    config = load_config()

    if args.mode == "ingest":
        from src.ingest import load_and_split_documents
        from src.embed import create_vector_store

        print("\n=== INGESTION ===")
        chunks = load_and_split_documents(
            raw_data_path=config["paths"]["raw_data"],
            chunk_size=config["parameters"]["chunk_size"],
            chunk_overlap=config["parameters"]["chunk_overlap"],
        )
        if chunks:
            create_vector_store(chunks)
        return

    from src.generator import RAGPipeline

    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"[Error] Could not start the pipeline (did you run --mode ingest first?): {e}")
        return

    def answer(question: str):
        result = pipeline.ask(question)
        print(f"\n{result.answer}")
        if result.sources:
            print(f"\nSources: {result.format_sources()}")

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
