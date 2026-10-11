# src/generator.py

import os
import warnings
from dataclasses import dataclass
from typing import List

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging

from src.embed import build_embeddings, load_config, load_vector_store

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity_error()

SYSTEM_PROMPT = (
    "You are a strict assistant that answers questions about a document. Use ONLY facts "
    "that are explicitly written in the provided context. Never add details from your own "
    "knowledge or imagination, and never guess. If the context does not explicitly answer "
    "the question, reply exactly: 'I do not know based on the provided documents.' "
    "Otherwise answer in at most three sentences."
)


@dataclass
class RAGAnswer:
    answer: str
    sources: List[Document]

    def format_sources(self) -> str:
        seen = []
        for doc in self.sources:
            ref = f"{doc.metadata.get('source', '?')} (page {doc.metadata.get('page', '?')})"
            if ref not in seen:
                seen.append(ref)
        return ", ".join(seen)


class RAGPipeline:
    """
    Loads the embedding model, vector store and LLM once, so many questions can be
    answered without paying the model start-up cost each time.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        models, params = self.config["models"], self.config["parameters"]

        print("[Pipeline] Loading vector store and embedding model...")
        self.db = load_vector_store(self.config, build_embeddings(self.config))

        print(f"[Pipeline] Loading LLM: {models['llm']}")
        self.tokenizer = AutoTokenizer.from_pretrained(models["llm"])
        self.model = AutoModelForCausalLM.from_pretrained(models["llm"], **self._model_kwargs())
        self.model.eval()

        self.top_k = params.get("top_k", 3)
        self.max_new_tokens = params.get("max_new_tokens", 256)
        self.temperature = params.get("temperature", 0.0)
        self.max_distance = params.get("max_distance")

    def _model_kwargs(self) -> dict:
        wants_4bit = self.config["models"].get("quantization") == "4bit"
        if wants_4bit and torch.cuda.is_available():
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            }
        if wants_4bit:
            print("[Pipeline] 4-bit quantization needs a CUDA GPU; falling back to unquantized weights.")
        return {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32, "device_map": "auto"}

    def ask(self, query: str) -> RAGAnswer:
        scored = self.db.similarity_search_with_score(query, k=self.top_k)
        # Chroma returns a distance (lower = more similar). Chunks beyond max_distance
        # are treated as irrelevant so the LLM is never asked to answer from noise.
        docs = [d for d, dist in scored if self.max_distance is None or dist <= self.max_distance]
        if not docs:
            return RAGAnswer("I do not know based on the provided documents.", [])

        context = "\n---\n".join(doc.page_content for doc in docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **gen_kwargs)

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return RAGAnswer(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), docs)


def run_rag_pipeline(query: str, config_path: str = "config.yaml") -> str:
    """One-shot convenience wrapper (loads everything per call; prefer RAGPipeline)."""
    return RAGPipeline(config_path).ask(query).answer


if __name__ == "__main__":
    result = RAGPipeline().ask("What does the narrator say about the character's costume?")
    print(result.answer)
    print("Sources:", result.format_sources())
