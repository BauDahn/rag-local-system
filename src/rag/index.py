# src/rag/index.py

import json
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.rag.ingest import hash_pdfs

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MANIFEST_FILE = "manifest.json"
CHUNKS_FILE = "chunks.json"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"\w+")


class IndexMismatchError(RuntimeError):
    """The persisted index was built with settings that differ from config.yaml."""


@dataclass
class Index:
    """Everything the retriever needs: dense store, keyword index and the chunk list."""

    db: Chroma
    bm25: BM25Okapi
    chunks: List[Document]
    embeddings: HuggingFaceEmbeddings
    query_prefix: str


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def query_prefix_for(model_name: str) -> str:
    """BGE English retrievers expect an instruction on queries (not on passages)."""
    name = model_name.lower()
    return BGE_QUERY_PREFIX if "bge" in name and "-en" in name else ""


def build_embeddings(config: dict) -> HuggingFaceEmbeddings:
    """Embedding model shared by ingestion and querying (they must match)."""
    return HuggingFaceEmbeddings(
        model_name=config["models"]["embedding"],
        model_kwargs={"device": config["models"].get("embedding_device", "cpu")},
        encode_kwargs={"normalize_embeddings": True},  # unit vectors -> clean cosine similarity
    )


def _manifest_for(config: dict, n_chunks: int) -> dict:
    params = config["parameters"]
    return {
        "embedding_model": config["models"]["embedding"],
        "chunk_size": params["chunk_size"],
        "chunk_overlap": params["chunk_overlap"],
        "min_letters": params.get("min_letters", 120),
        "pdf_hash": hash_pdfs(config["paths"]["raw_data"]),
        "n_chunks": n_chunks,
    }


def _clear_store(store_dir: Path) -> None:
    """Removes a previous index, including orphan Chroma segment folders. Refuses to touch
    a directory that does not look like one of ours."""
    if not store_dir.exists():
        return
    if any(store_dir.iterdir()) and not (store_dir / "chroma.sqlite3").exists():
        raise RuntimeError(f"{store_dir} is not empty and does not look like a vector store; refusing to wipe it.")
    shutil.rmtree(store_dir)


def create_index(chunks: List[Document], config: dict) -> Index:
    """
    Embeds the chunks into a cosine Chroma collection, saves the chunk list for BM25
    and writes a manifest. Any previous index is dropped, so re-running never duplicates.
    """
    store_dir = Path(config["paths"]["vector_store"])
    _clear_store(store_dir)
    store_dir.mkdir(parents=True)

    chunks = [Document(page_content=c.page_content, metadata={**c.metadata, "chunk_id": i}) for i, c in enumerate(chunks)]
    embeddings = build_embeddings(config)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(store_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    with open(store_dir / CHUNKS_FILE, "w") as f:
        json.dump([{"text": c.page_content, "metadata": c.metadata} for c in chunks], f)
    with open(store_dir / MANIFEST_FILE, "w") as f:
        json.dump(_manifest_for(config, len(chunks)), f, indent=2)

    print(f"[Index] Stored {len(chunks)} chunks in {store_dir}")
    return _assemble(db, chunks, embeddings, config)


def check_manifest(config: dict) -> None:
    """Raises IndexMismatchError if the stored index no longer matches config.yaml."""
    store_dir = Path(config["paths"]["vector_store"])
    manifest_path = store_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise IndexMismatchError(f"No index found in {store_dir}. Run `python main.py --mode ingest` first.")

    with open(manifest_path) as f:
        stored = json.load(f)
    params = config["parameters"]
    expected = {
        "embedding_model": config["models"]["embedding"],
        "chunk_size": params["chunk_size"],
        "chunk_overlap": params["chunk_overlap"],
        "min_letters": params.get("min_letters", 120),
    }
    diffs = [f"{k}: index={stored.get(k)!r} config={v!r}" for k, v in expected.items() if stored.get(k) != v]
    if diffs:
        raise IndexMismatchError(
            "The index was built with different settings (" + "; ".join(diffs) + "). Re-run `--mode ingest`."
        )


def load_index(config: dict) -> Index:
    """Opens the persisted index after checking that it matches the configuration."""
    check_manifest(config)
    store_dir = Path(config["paths"]["vector_store"])
    embeddings = build_embeddings(config)
    db = Chroma(
        persist_directory=str(store_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    with open(store_dir / CHUNKS_FILE) as f:
        chunks = [Document(page_content=row["text"], metadata=row["metadata"]) for row in json.load(f)]
    return _assemble(db, chunks, embeddings, config)


def _assemble(db: Chroma, chunks: List[Document], embeddings: HuggingFaceEmbeddings, config: dict) -> Index:
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    return Index(db, bm25, chunks, embeddings, query_prefix_for(config["models"]["embedding"]))
