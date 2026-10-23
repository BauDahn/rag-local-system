import json

import httpx
import pytest

from src.rag.config import ROOT, load_config
from src.rag.evaluate import DEFAULT_EVAL_SET, resolve_eval_path
from src.rag.llm import ANSWER_SCHEMA, LLMError, OllamaLLM, OpenAICompatLLM, create_llm


def make_config(**models):
    return {
        "models": {"llm": "m", **models},
        "parameters": {"temperature": 0, "max_new_tokens": 50},
        "llm_server": {"base_url": "http://srv/v1/", "api_key": "k"},
    }


def llm_with(handler, config=None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatLLM(config or make_config(), client=client)


def ok(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_openai_request_and_parse():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return ok('{"answer": "x", "citations": [1], "answerable": true}')

    out = llm_with(handler).chat_json([{"role": "user", "content": "hi"}])
    assert out["answer"] == "x"
    assert seen["url"] == "http://srv/v1/chat/completions"
    rf = seen["body"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == ANSWER_SCHEMA
    assert seen["body"]["model"] == "m" and seen["body"]["max_tokens"] == 50


def test_openai_api_key_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_LLM_API_KEY", "secret")
    llm = OpenAICompatLLM(make_config())
    assert llm.client.headers["authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    "response, text",
    [
        (httpx.Response(404), "not found"),
        (httpx.Response(401), "API key"),
        (httpx.Response(500, text="boom"), "500"),
        (ok("not json"), "valid JSON"),
        (ok('{"answer": "cut', finish="length"), "truncated"),
        (httpx.Response(200, json={"oops": 1}), "Unexpected"),
    ],
)
def test_openai_errors(response, text):
    with pytest.raises(LLMError, match=text):
        llm_with(lambda r: response).chat_json([])


def test_openai_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Could not reach"):
        llm_with(handler).chat_json([])


def test_factory():
    assert isinstance(create_llm(make_config()), OllamaLLM)
    assert isinstance(create_llm(make_config(llm_backend="ollama")), OllamaLLM)
    assert isinstance(create_llm(make_config(llm_backend="openai")), OpenAICompatLLM)
    with pytest.raises(LLMError, match="Unknown"):
        create_llm(make_config(llm_backend="nope"))


def test_ollama_extra_options():
    class FakeClient:
        def chat(self, **kw):
            self.kw = kw
            return {"message": {"content": "{}"}}

    cfg = make_config()
    cfg["ollama"] = {"num_ctx": 8192, "keep_alive": "1h"}
    llm = OllamaLLM(cfg)
    llm.client = FakeClient()
    llm.chat_json([])
    assert llm.client.kw["options"]["num_ctx"] == 8192 and llm.client.kw["keep_alive"] == "1h"
    plain = OllamaLLM(make_config())
    plain.client = FakeClient()
    plain.chat_json([])
    assert "num_ctx" not in plain.client.kw["options"] and "keep_alive" not in plain.client.kw


def test_paths_resolve_against_project_root(tmp_path, monkeypatch):
    sub = tmp_path / "configs"
    sub.mkdir()
    cfg = sub / "p.yaml"
    cfg.write_text(
        "models: {llm: m}\npaths: {raw_data: data/raw, vector_store: /abs/db, eval_set: data/e.yaml}\n"
    )
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    paths = load_config(str(cfg))["paths"]
    assert paths["raw_data"] == str(ROOT / "data" / "raw")
    assert paths["vector_store"] == "/abs/db"
    assert paths["eval_set"] == str(ROOT / "data" / "e.yaml")


def test_api_key_env_in_config(monkeypatch, tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text("models: {llm: m}\npaths: {raw_data: x}\n")
    monkeypatch.setenv("RAG_LLM_API_KEY", "abc")
    assert load_config(str(cfg))["llm_server"]["api_key"] == "abc"


def test_resolve_eval_path():
    assert resolve_eval_path({"paths": {"eval_set": "a"}}) == "a"
    assert resolve_eval_path({"paths": {}}) == DEFAULT_EVAL_SET
    assert resolve_eval_path({"paths": {"eval_set": "a"}}, "b") == "b"


def test_shipped_profiles_load():
    for name in ["config.yaml", "configs/dgt.yaml", "configs/empresa-multi-gpu.example.yaml"]:
        cfg = load_config(str(ROOT / name))
        assert cfg["paths"]["raw_data"].startswith(str(ROOT))
        assert (ROOT / "src/rag/prompts" / f"{cfg['prompts']['template']}.md").exists()


def test_answer_schema_bounds_citations():
    from src.rag.llm import answer_schema

    schema = answer_schema(4)
    cit = schema["properties"]["citations"]
    assert cit["maxItems"] == 4 and cit["items"]["enum"] == [1, 2, 3, 4]
    assert list(schema["properties"]) == ["answer", "citations", "answerable"]  # field order is load-bearing
    assert ANSWER_SCHEMA["properties"]["citations"].get("maxItems") is None  # original untouched
