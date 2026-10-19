import json

import pytest

from src.rag.index import (
    BGE_QUERY_PREFIX,
    IndexMismatchError,
    _clear_store,
    check_manifest,
    query_prefix_for,
    tokenize,
)


def make_config(tmp_path):
    return {
        "models": {"embedding": "BAAI/bge-small-en-v1.5"},
        "parameters": {"chunk_size": 512, "chunk_overlap": 50, "min_letters": 120},
        "paths": {"raw_data": str(tmp_path / "raw"), "vector_store": str(tmp_path / "db")},
    }


def write_manifest(config, **overrides):
    store = config["paths"]["vector_store"]
    manifest = {"embedding_model": "BAAI/bge-small-en-v1.5", "chunk_size": 512, "chunk_overlap": 50, "min_letters": 120}
    manifest.update(overrides)
    import os

    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("The Fox, the B-612!") == ["the", "fox", "the", "b", "612"]


def test_query_prefix_only_for_english_bge():
    assert query_prefix_for("BAAI/bge-small-en-v1.5") == BGE_QUERY_PREFIX
    assert query_prefix_for("BAAI/bge-m3") == ""
    assert query_prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_manifest_missing_asks_for_ingest(tmp_path):
    with pytest.raises(IndexMismatchError, match="--mode ingest"):
        check_manifest(make_config(tmp_path))


def test_manifest_match_passes(tmp_path):
    config = make_config(tmp_path)
    write_manifest(config)
    check_manifest(config)


def test_manifest_detects_embedding_model_change(tmp_path):
    config = make_config(tmp_path)
    write_manifest(config, embedding_model="other/model")
    with pytest.raises(IndexMismatchError, match="embedding_model"):
        check_manifest(config)


def test_clear_store_refuses_foreign_directory(tmp_path):
    foreign = tmp_path / "important"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(RuntimeError):
        _clear_store(foreign)
    assert (foreign / "notes.txt").exists()


def test_clear_store_removes_chroma_directory(tmp_path):
    store = tmp_path / "db"
    (store / "orphan-uuid").mkdir(parents=True)
    (store / "chroma.sqlite3").write_text("")
    _clear_store(store)
    assert not store.exists()
