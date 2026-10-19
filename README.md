# Local RAG System

> 🚧 **Work in progress.** This is an actively developed learning project. Ingestion, hybrid retrieval, reranking and evaluation work; see [Known limitations](#known-limitations) and the [Roadmap](#roadmap).

A fully **local, private Retrieval-Augmented Generation (RAG)** pipeline for asking questions about your own PDF documents. No API keys, no cloud services: documents, embeddings, reranker and language model all stay on your machine.

Answers are grounded in the retrieved text and come with **verified source citations** (file and page): the model must cite the numbered fragments it used, and citations that don't point to a fragment it was given are discarded. When the documents don't contain the answer, the system says *"I do not know"* instead of inventing one.

## Documentation

- [COMO_FUNCIONA.md](COMO_FUNCIONA.md): beginner-friendly walkthrough of how a RAG works, with a glossary and exercises (in Spanish).
- [arquitectura.md](arquitectura.md): modules, data flow, design decisions and how to extend the system (in Spanish).

## How it works

```
 PDFs ─► ingest ─► chunks + metadata ─► ChromaDB (cosine) + BM25 index + manifest.json

 question ─► dense top-20 ─┐
             BM25  top-20 ─┴─► RRF fusion ─► cross-encoder rerank ─► top-4
                                                                       │
                          numbered fragments "[1] file, p.X" + prompt template
                                                                       │
                     Ollama LLM  ─►  JSON {answerable, answer, citations}
                                                                       │
                              citation check ─► answer + cited sources
```

| Stage | Implementation |
|---|---|
| PDF parsing | `pypdf`, page by page |
| Chunking | `RecursiveCharacterTextSplitter` (512 chars, 50 overlap); each chunk keeps its `source` and `page` |
| Dense retrieval | `BAAI/bge-small-en-v1.5` (query instruction prefix applied), ChromaDB with cosine distance |
| Keyword retrieval | BM25 (`rank-bm25`), good at exact names and numbers |
| Fusion | Reciprocal Rank Fusion of both rankings |
| Reranking | `BAAI/bge-reranker-base` cross-encoder re-scores the fused candidates |
| Generation | Any [Ollama](https://ollama.com) model (default `qwen2.5:3b-instruct`), JSON output constrained by a schema |
| Prompts | Plain-text templates in `src/rag/prompts/`, not hardcoded |
| Index safety | `manifest.json` records the embedding model and chunking settings; querying with a mismatching config fails with a clear message |

## Project structure

```
├── main.py                  # CLI entry point (ingest / query / eval)
├── config.yaml              # Models, retrieval, prompts and paths
├── pyproject.toml           # Dependencies
├── src/rag/
│   ├── config.py            # Config loading; paths resolved from the project root
│   ├── ingest.py            # PDF loading and chunking with metadata
│   ├── index.py             # Chroma + BM25 index, manifest, embeddings
│   ├── retrieve.py          # dense / hybrid / hybrid+rerank retrieval
│   ├── llm.py               # Ollama client with structured output
│   ├── pipeline.py          # RAGPipeline: retrieve -> prompt -> LLM -> citation check
│   ├── evaluate.py          # Retrieval ablation and answer evaluation
│   └── prompts/qa.md        # System and user prompt templates
├── tests/                   # pytest suite (no models needed)
└── data/
    ├── raw/                 # Put your PDFs here
    └── eval_set.yaml        # Example evaluation questions (written for The Little Prince)
```

## Getting started

```bash
git clone https://github.com/BauDahn/rag-local-system.git && cd rag-local-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Install [Ollama](https://ollama.com/download), start it (`ollama serve`, or the system service) and pull the model from `config.yaml`:

```bash
ollama pull qwen2.5:3b-instruct
```

1. Put one or more PDFs in `data/raw/`. No documents are bundled with the repo (PDFs there are git-ignored). `data/eval_set.yaml` was written for an English edition of *The Little Prince*; supply your own legally obtained copy, or replace the evaluation questions with ones about your own documents.
2. Build the index. Re-running replaces the previous index, so nothing is duplicated:
   ```bash
   python main.py --mode ingest
   ```
3. Ask questions:
   ```bash
   # One-shot
   python main.py --mode query --question "What does the prince look like?"

   # Interactive session (models are loaded once and reused)
   python main.py --mode query
   ```

### Hardware notes

- The default configuration targets a small GPU (about 4 GB of VRAM): a 3B model in Ollama, with the embedder and reranker on CPU. With more VRAM, pull a larger model (e.g. `qwen2.5:7b-instruct`, `llama3.1:8b`), change `models.llm`, and optionally set `embedding_device` / `reranker_device` to `cuda`.
- The first run downloads the embedding and reranker models from Hugging Face; after that retrieval works offline.

## Configuration

Everything is controlled from `config.yaml`: model names and devices, chunking, `k_candidates` / `top_k`, `rrf_k`, `min_rerank_score`, the prompt template and fallback text, and paths. Prompts live in `src/rag/prompts/<name>.md` (sections `---system---` and `---user---`); select one with `prompts.template`. To work with non-English documents, switch the embedding model to a multilingual one (e.g. `BAAI/bge-m3`) and re-run ingestion; the manifest stops you from querying an index built with a different model.

## Tests and evaluation

```bash
pytest                                  # unit tests, no models required
python main.py --mode eval              # retrieval ablation (no LLM needed)
python main.py --mode eval --answers    # also: refusals and faithfulness (needs Ollama)
```

The retrieval evaluation runs the same questions through dense, hybrid and hybrid+rerank retrieval and reports **hit rate@k** (is a relevant chunk in the top-k?) and **MRR** (how high does the first one rank?). Keywords match whole words, so "boa" does not match "boat". With `--answers`, questions marked `answerable: false` must be refused, and an LLM judge checks that each answer is supported by the fragments it cited.

Result on the author's test document (*The Little Prince*, 218 chunks, `top_k=4`, 9 answerable questions; the book itself is not included in this repo):

| Retrieval | Hit rate@4 | MRR |
|---|---|---|
| Dense only | 0.89 | 0.78 |
| Hybrid (dense + BM25) | 0.89 | 0.69 |
| Hybrid + rerank | 0.89 | 0.81 |

The reranker gives the best MRR (relevant chunks rank higher) without changing the hit rate. Hybrid search alone doesn't help on this set, which is expected: its benefit shows on exact names and numbers, and this question set has few of them. With only 9 questions on one book, these numbers are indicative, not conclusive; add questions about your own documents to benchmark them.

Answer evaluation (`--answers`) with `qwen2.5:3b-instruct` on the same set:

| Metric | Score |
|---|---|
| Refusal accuracy (2 unanswerable questions) | 1.00 |
| Answerable questions refused | 2 of 9 |
| Faithfulness (judged by the same 3B model) | 0.33 |

The faithfulness number should be read with care: the judge is the same small model that wrote the answers, and it is unreliable (for instance it rejected a correct one-word answer). Use a larger model as judge, or review answers manually, before drawing conclusions.

## Design decisions

- **Local-first**: suited to confidential documents where sending data to a third-party API isn't an option.
- **Ollama instead of in-process models**: the LLM is a swappable HTTP service, so changing model is a one-line config change and the code doesn't depend on a `transformers` version.
- **Metadata preserved end to end**: chunks are split per page so citations are accurate and chunks never straddle two documents.
- **Verifiable citations**: the model sees numbered, labelled fragments and returns the numbers it used; only valid ones are shown as sources, and none are shown when it refuses.
- **Structured output**: a JSON schema forces `{answerable, answer, citations}`, so abstention and citations don't depend on parsing free text.
- **Idempotent, versioned index**: ingestion rebuilds everything (including stale Chroma folders) and writes a manifest; a mismatching config is detected instead of silently returning bad results.
- **Deterministic by default**: `temperature: 0` to favor faithful answers.
- **Noise filtering**: near-empty chunks (title pages, page numbers) are dropped at ingestion.

## Known limitations

- **Small-model comprehension.** A 3B model sometimes refuses questions its context answers, can misattribute who said what, and is sensitive to prompt wording. The JSON schema lists `answer` before `answerable` because forcing the decision first made refusals much more frequent. A larger model (7-8B, partly offloaded to CPU on a 4 GB GPU) is the most likely improvement and has not been benchmarked here yet.
- **No relevance gate yet.** The old distance threshold was removed with the move to cosine distance and reranking; out-of-scope questions are now refused by the model (`answerable: false`). `min_rerank_score` can add a cheap pre-filter once calibrated on your documents.
- **Small evaluation set.** See above.

## Roadmap

- [x] Reranking with a cross-encoder
- [x] Hybrid retrieval (BM25 + dense)
- [x] Answer-quality evaluation (refusals + LLM-judged faithfulness)
- [ ] Calibrate `min_rerank_score` and a larger evaluation set
- [ ] Query rewriting and parent-document retrieval
- [ ] Support for more formats (DOCX, Markdown, HTML)
- [ ] Simple web UI

## License

Released under the [MIT License](LICENSE).

## Tech stack

Python · Ollama · PyTorch · sentence-transformers · rank-bm25 · LangChain (text splitters, integrations) · ChromaDB · pypdf
