# src/rag/llm.py

import json
import os
from typing import List, Optional

import httpx
import ollama

# Property order matters: the model generates the fields in this order, and small models
# refuse far too often when they must commit to `answerable` before writing the answer.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "citations", "answerable"],
}


def answer_schema(n_fragments: int) -> dict:
    """ANSWER_SCHEMA with `citations` limited to the fragment numbers that exist.

    An unbounded integer array lets small models fall into a loop of repeated numbers until
    they hit the token cap, which leaves the JSON truncated and unparseable.
    """
    schema = json.loads(json.dumps(ANSWER_SCHEMA))
    schema["properties"]["citations"] = {
        "type": "array",
        "items": {"type": "integer", "enum": list(range(1, n_fragments + 1))},
        "maxItems": n_fragments,
    }
    return schema


class LLMError(RuntimeError):
    """The LLM server is unreachable, the model is missing, or the reply could not be parsed."""


class OllamaLLM:
    """Backend `ollama`: local Ollama daemon (default; good for a single small GPU)."""

    def __init__(self, config: dict):
        self.model = config["models"]["llm"]
        self.temperature = config["parameters"].get("temperature", 0.0)
        self.max_new_tokens = config["parameters"].get("max_new_tokens", 300)
        ollama_cfg = config.get("ollama") or {}
        self.num_ctx = ollama_cfg.get("num_ctx")  # None = Ollama's default context window
        self.keep_alive = ollama_cfg.get("keep_alive")  # None = Ollama's default (5m)
        self.client = ollama.Client(host=ollama_cfg.get("host"), timeout=ollama_cfg.get("timeout", 120))

    def chat_json(self, messages: List[dict], schema: dict = ANSWER_SCHEMA) -> dict:
        """Sends a chat and returns the reply parsed as JSON constrained by `schema`."""
        options = {"temperature": self.temperature, "num_predict": self.max_new_tokens}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        kwargs = {"keep_alive": self.keep_alive} if self.keep_alive is not None else {}
        try:
            response = self.client.chat(model=self.model, messages=messages, format=schema, options=options, **kwargs)
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


class OpenAICompatLLM:
    """
    Backend `openai`: any server speaking the OpenAI chat API (vLLM with tensor parallelism
    across GPUs, llama.cpp server, SGLang, LM Studio...). Plain HTTP via httpx, which is
    already installed as a dependency of the `ollama` package.
    """

    def __init__(self, config: dict, client: Optional[httpx.Client] = None):
        self.model = config["models"]["llm"]
        self.temperature = config["parameters"].get("temperature", 0.0)
        self.max_new_tokens = config["parameters"].get("max_new_tokens", 300)
        server = config.get("llm_server") or {}
        self.base_url = str(server.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        api_key = os.environ.get("RAG_LLM_API_KEY") or server.get("api_key") or "EMPTY"
        self.client = client or httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=server.get("timeout", 120)
        )

    def chat_json(self, messages: List[dict], schema: dict = ANSWER_SCHEMA) -> dict:
        """Sends a chat and returns the reply parsed as JSON constrained by `schema`."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = self.client.post(url, json=payload)
        except Exception as e:  # connection refused, timeouts, ...
            raise LLMError(f"Could not reach the LLM server at {self.base_url} (is it running?): {e}") from e

        if response.status_code == 404:
            raise LLMError(
                f"Model {self.model!r} or endpoint {url} not found on the server. "
                "Check models.llm (must match the served model name) and llm_server.base_url (usually ends in /v1)."
            )
        if response.status_code in (401, 403):
            raise LLMError("The LLM server rejected the API key. Set llm_server.api_key or RAG_LLM_API_KEY.")
        if response.status_code >= 400:
            raise LLMError(f"LLM server error {response.status_code}: {response.text[:300]}")

        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected reply from the LLM server: {e}") from e
        if choice.get("finish_reason") == "length":
            raise LLMError("The reply was truncated (max_new_tokens too low?); it is not valid JSON.")
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise LLMError(f"The model did not return valid JSON: {e}") from e


BACKENDS = {"ollama": OllamaLLM, "openai": OpenAICompatLLM}


def create_llm(config: dict):
    """Factory: builds the backend selected by `models.llm_backend` (default `ollama`)."""
    name = str(config["models"].get("llm_backend", "ollama")).lower()
    if name not in BACKENDS:
        raise LLMError(f"Unknown models.llm_backend {name!r}; use one of: {', '.join(BACKENDS)}")
    return BACKENDS[name](config)
