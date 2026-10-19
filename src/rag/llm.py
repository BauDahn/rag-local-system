# src/rag/llm.py

import json
from typing import List

import ollama

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "citations"],
}


class LLMError(RuntimeError):
    """Ollama is unreachable, the model is missing, or the reply could not be parsed."""


class OllamaLLM:
    def __init__(self, config: dict):
        self.model = config["models"]["llm"]
        self.temperature = config["parameters"].get("temperature", 0.0)
        self.max_new_tokens = config["parameters"].get("max_new_tokens", 300)
        ollama_cfg = config.get("ollama", {})
        self.client = ollama.Client(host=ollama_cfg.get("host"), timeout=ollama_cfg.get("timeout", 120))

    def chat_json(self, messages: List[dict], schema: dict = ANSWER_SCHEMA) -> dict:
        """Sends a chat and returns the reply parsed as JSON constrained by `schema`."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=schema,
                options={"temperature": self.temperature, "num_predict": self.max_new_tokens},
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMError(f"Model {self.model!r} is not installed. Run: ollama pull {self.model}") from e
            raise LLMError(f"Ollama error: {e}") from e
        except Exception as e:  # connection refused, timeouts, ...
            raise LLMError(f"Could not reach Ollama (is `ollama serve` running?): {e}") from e

        try:
            return json.loads(response["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise LLMError(f"The model did not return valid JSON: {e}") from e
