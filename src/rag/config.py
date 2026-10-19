# src/rag/config.py

import os
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Loads config.yaml and resolves every entry under `paths` against the project root,
    so the code works no matter which directory it is launched from.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    base = path.resolve().parent
    # Handy for comparing models without editing the file: RAG_LLM=qwen2.5:7b-instruct python main.py ...
    if os.environ.get("RAG_LLM"):
        config["models"]["llm"] = os.environ["RAG_LLM"]

    config["paths"] = {name: str((base / value).resolve()) for name, value in config["paths"].items()}
    return config
