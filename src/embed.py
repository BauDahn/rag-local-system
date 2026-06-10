# src/embed.py

import os
import yaml
import warnings
from typing import List
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_vector_store(chunks: List[str], config_path: str = "config.yaml") -> Chroma:
    """
    Transforms text chunks into vector embeddings using a Hugging Face model
    and persists them into a local ChromaDB instance.
    """
    # Load configuration parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["models"]["embedding"]
    persist_directory = config["paths"]["vector_store"]

    # Configure the model to run on CPU/GPU efficiently
    model_kwargs = {'device': 'cpu'} # You can change to 'cuda' later if you have your GPU configured
    encode_kwargs = {'normalize_embeddings': True} # Scientific norm for clean cosine similarity matches

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create the vector database and save it directly to the local disk
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_db

if __name__ == "__main__":
    # Test the embedding engine using the script we built previously
    print("Testing Embedding Module locally...")
    from src.ingest import load_and_split_documents

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Fetch chunks using your working ingest module
    chunks = load_and_split_documents(
        raw_data_path=cfg["paths"]["raw_data"],
        chunk_size=cfg["parameters"]["chunk_size"],
        chunk_overlap=cfg["parameters"]["chunk_overlap"]
    )

    if chunks:
        # 2. Vectorize and save them locally
        db = create_vector_store(chunks)
        
        # 3. Quick semantic test query to prove it works mathematically
        print("\n[Test] Running a quick semantic search validation...")
        test_query = "What does the prince look like?"
        docs = db.similarity_search(test_query, k=1)
        
        print(f"Query: '{test_query}'")
        print(f"Most similar retrieved text chunk:\n{docs[0].page_content}")