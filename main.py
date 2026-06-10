# main.py

import argparse
import yaml
from src.ingest import load_and_split_documents
from src.embed import create_vector_store
from src.generator import run_rag_pipeline

def main():
    # Setup CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="Local Guarded RAG System - Secure On-Premise Document Intelligence"
    )
    
    # Define execution modes
    parser.add_argument(
        "--mode", 
        choices=["ingest", "query"], 
        required=True,
        help="Choose 'ingest' to process raw data directory or 'query' to prompt the system."
    )
    
    # Optional custom query flag
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="The technical question to pass into the system when running in query mode."
    )

    args = parser.parse_args()

    # Load parameters to pass across modules
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.mode == "ingest":
        print("\n=== STARTING DATA INGESTION WORKFLOW ===")
        chunks = load_and_split_documents(
            raw_data_path=config["paths"]["raw_data"],
            chunk_size=config["parameters"]["chunk_size"],
            chunk_overlap=config["parameters"]["chunk_overlap"]
        )
        if chunks:
            create_vector_store(chunks)
        print("=========================================\n")

    elif args.mode == "query":
        # Fallback if no specific question flag was provided
        query_text = args.question
        if not query_text:
            query_text = input("\nEnter your document query: ")
        
        print("\n=== INVOKING RETRIEVAL INFERENCE ENGINE ===")
        try:
            answer = run_rag_pipeline(query_text)
            print("\n---------------- SYSTEM RESPONSE ----------------")
            print(answer)
            print("-------------------------------------------------")
        except Exception as e:
            print(f"[Runtime Error]: Pipeline inference failed -> {e}")
        print("===========================================\n")

if __name__ == "__main__":
    main()