# src/embed.py

import warnings
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_embeddings(config: dict) -> HuggingFaceEmbeddings:
    """Embedding model shared by ingestion and querying (they must match)."""
    return HuggingFaceEmbeddings(
        model_name=config["models"]["embedding"],
        model_kwargs={"device": config["models"].get("embedding_device", "cpu")},
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> clean cosine similarity
    )


def load_vector_store(config: dict, embeddings: HuggingFaceEmbeddings = None) -> Chroma:
    """Opens the persisted ChromaDB collection."""
    return Chroma(
        persist_directory=config["paths"]["vector_store"],
        embedding_function=embeddings or build_embeddings(config),
    )


def create_vector_store(chunks: List[Document], config_path: str = "config.yaml") -> Chroma:
    """
    Embeds the chunks and persists them in ChromaDB. Any existing collection is
    dropped first, so re-running ingestion never duplicates data.
    """
    config = load_config(config_path)
    embeddings = build_embeddings(config)

    existing = load_vector_store(config, embeddings)
    existing.delete_collection()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config["paths"]["vector_store"],
    )
    print(f"[Embed] Stored {len(chunks)} chunks in {config['paths']['vector_store']}")
    return vector_db


if __name__ == "__main__":
    from src.ingest import load_and_split_documents

    cfg = load_config()
    docs = load_and_split_documents(
        cfg["paths"]["raw_data"],
        cfg["parameters"]["chunk_size"],
        cfg["parameters"]["chunk_overlap"],
    )
    if docs:
        db = create_vector_store(docs)
        query = "What does the prince look like?"
        hit = db.similarity_search(query, k=1)[0]
        print(f"\nQuery: {query}\nTop chunk {hit.metadata}:\n{hit.page_content}")
