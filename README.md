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
├── main.py                  # CLI entry point (ingest / query / eval / ui)
├── config.yaml              # Default profile: models, retrieval, prompts and paths
├── configs/                 # Other profiles (dgt.yaml, empresa-multi-gpu.example.yaml); use with --config
├── docs/despliegue-modelos-grandes.md  # Running 32B-70B models on several GPUs (vLLM)
├── pyproject.toml           # Dependencies
├── src/rag/
│   ├── config.py            # Config loading; paths resolved from the project root
│   ├── ingest.py            # PDF loading and chunking with metadata
│   ├── index.py             # Chroma + BM25 index, manifest, embeddings
│   ├── retrieve.py          # dense / hybrid / hybrid+rerank retrieval
│   ├── llm.py               # LLM backends (Ollama / OpenAI-compatible) + factory
│   ├── pipeline.py          # RAGPipeline: retrieve -> prompt -> LLM -> citation check
│   ├── ui.py                # Gradio web UI
│   ├── evaluate.py          # Retrieval ablation and answer evaluation
│   └── prompts/             # qa.md (English), qa_es.md (Spanish)
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
4. Or use the web UI (optional dependency: `pip install -e ".[ui]"`). It listens only on `127.0.0.1:7860` and shows the answer plus the cited sources (file, page, fragment); it reports clearly if Ollama is not responding or the index does not match the config:
   ```bash
   python main.py --mode ui
   ```

### Hardware notes

- The default configuration targets a small GPU (about 4 GB of VRAM): a 3B model in Ollama, with the embedder and reranker on CPU. With more VRAM, pull a larger model (e.g. `qwen2.5:7b-instruct`, `llama3.1:8b`), change `models.llm`, and optionally set `embedding_device` / `reranker_device` to `cuda`.
- The first run downloads the embedding and reranker models from Hugging Face; after that retrieval works offline.

## Configuration

Everything is controlled from `config.yaml` (default profile). Other profiles live in `configs/` and are selected with `--config configs/dgt.yaml` in every mode (`--eval-set` overrides `paths.eval_set`). Relative paths are resolved against the project root. The LLM backend is `models.llm_backend: ollama | openai`; `openai` talks to any OpenAI-compatible server (vLLM with tensor parallelism, llama.cpp server), so larger models on several GPUs need only configuration: see [docs/despliegue-modelos-grandes.md](docs/despliegue-modelos-grandes.md). Settings: model names and devices, chunking, `k_candidates` / `top_k`, `rrf_k`, `min_rerank_score`, the prompt template and fallback text, and paths. Prompts live in `src/rag/prompts/<name>.md` (sections `---system---` and `---user---`); select one with `prompts.template`. To work with non-English documents, switch the embedding model to a multilingual one (e.g. `BAAI/bge-m3`) and re-run ingestion; the manifest stops you from querying an index built with a different model.

## Tests and evaluation

```bash
pytest                                  # unit tests, no models required
python main.py --mode eval              # retrieval ablation (no LLM needed)
python main.py --mode eval --answers    # also: refusals and answer accuracy (needs Ollama)
RAG_LLM=qwen2.5:7b-instruct python main.py --mode eval --answers   # compare models without editing config.yaml
```

`data/eval_set.yaml` has 46 questions about *The Little Prince*: 38 answerable and 8 the book cannot answer. Each answerable one lists `keywords` (to check retrieval) and `expected` words (to check the answer); both match whole words, so "boa" does not match "boat".

- **Retrieval** runs every question through dense, hybrid and hybrid+rerank and reports **hit rate@k** (is a relevant chunk in the top-k?) and **MRR** (how high does the first one rank?).
- **Answers** (`--answers`) report refusal accuracy (unanswerable questions must be refused), answer accuracy (the answer contains an expected word), the share of answerable questions wrongly refused, and a self-judged faithfulness score.

Results on the author's test document (218 chunks, `top_k=4`; the book itself is not included in this repo):

| Retrieval (38 questions) | Hit rate@4 | MRR |
|---|---|---|
| Dense only | 0.71 | 0.57 |
| Hybrid (dense + BM25) | 0.79 | 0.65 |
| Hybrid + rerank | 0.87 | 0.76 |

Each stage helps: BM25 recovers passages with exact names and phrases, and the reranker pushes the right chunk higher. (An earlier 9-question set hid the hybrid gain; the larger set shows it.)

| Answers (46 questions) | `qwen2.5:3b-instruct` | `qwen2.5:7b-instruct` |
|---|---|---|
| Refusal accuracy (8 unanswerable) | 1.00 | 1.00 |
| Answer accuracy (38 answerable) | 0.74 | 0.76 |
| Answerable questions wrongly refused | 0.03 | 0.11 |
| Faithfulness (self-judged) | 0.54 | 0.79 |

The two models are effectively tied on accuracy, so the 3B model stays the default: it is faster and fits a 4 GB GPU (the 7B runs about half on CPU). Read these numbers with care:

- Answer accuracy is a keyword check and is conservative. Some answers marked wrong are arguable (for example "Good morning" as the echo's reply, or the stars as "wells with a rusty pulley", which is in the book), and others are real errors (the 3B model said the first thing the prince asked for was "his airplane").
- Faithfulness is judged by the same model that writes the answers, so it is a weak signal.
- It is one run per model with temperature 0, on a single book. Add questions about your own documents to benchmark them.

### Second corpus: a Spanish legal manual

To check that results are not specific to one novel, the same pipeline was run on the *Reglamento General de Circulación* (Spanish traffic regulations, 730 pages, downloaded from the BOE; not included in the repo) with `configs/dgt.yaml` (multilingual `bge-m3` embeddings and `bge-reranker-v2-m3`). Its 51 questions (42 answerable, 9 not) live in `data/corpora/dgt/eval_set.yaml`.

| Retrieval (42 questions, 557 chunks) | Hit rate@4 | MRR |
|---|---|---|
| Dense only | 1.00 | 0.93 |
| Hybrid (dense + BM25) | 1.00 | 0.95 |
| Hybrid + rerank | 1.00 | 0.96 |

| Answers (51 questions) | `qwen2.5:3b-instruct` | `qwen2.5:7b-instruct` |
|---|---|---|
| Refusal accuracy (9 unanswerable) | 1.00 | 1.00 |
| Answer accuracy (42 answerable) | 0.90 | 0.90 |
| Faithfulness (self-judged) | 0.38 | 0.88 |

Manuals are easier to retrieve from than a novel: every question has distinctive terms, so even dense retrieval finds the passage. Read the 1.00 with care: the questions and their `keywords` were written from the source text, which favors retrieval. The remaining answer errors are comprehension, not retrieval: the right passage was in the context but the 7B picked a neighbouring figure (80 km/h for urban motorways instead of 120 km/h; 0,15 mg/l for professional drivers instead of 0,25 mg/l for general drivers). That is the case for larger models, which the architecture now supports (see [docs/despliegue-modelos-grandes.md](docs/despliegue-modelos-grandes.md)); it has not been measured here because the author's GPU has 4 GB.

Two bugs found along the way that affect any small model: an unbounded `citations` array made the model loop on repeated numbers until the JSON was cut off (now bounded to the fragments that exist), and when it wanted to refuse the 7B wrote the word "answerable" inside `answer` (fixed by showing the exact refusal JSON in the Spanish prompt).

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

- **Small-model comprehension.** A 3B model sometimes refuses questions its context answers, can misattribute who said what, and is sensitive to prompt wording. The JSON schema lists `answer` before `answerable` because forcing the decision first made refusals much more frequent. Moving to a 7B model did not improve accuracy on the evaluation set (see above), so model size alone is not the fix; prompt wording and better evaluation are the next levers.
- **No relevance gate yet.** The old distance threshold was removed with the move to cosine distance and reranking; out-of-scope questions are now refused by the model (`answerable: false`). `min_rerank_score` is off: on the evaluation set the top reranker score is 0.00-0.09 for unanswerable questions but also below 0.05 for 7 of 38 answerable ones, so any threshold would wrongly refuse more than it saves.
- **Evaluation covers one book.** 46 questions are enough to compare retrieval strategies, but too few to separate models that differ by a few points.

## Roadmap

- [x] Reranking with a cross-encoder
- [x] Hybrid retrieval (BM25 + dense)
- [x] Answer-quality evaluation (refusals + LLM-judged faithfulness)
- [x] Larger evaluation set (46 questions) with keyword-based answer accuracy
- [x] Calibrate `min_rerank_score` using that set (result: leave it off, scores do not separate answerable from unanswerable questions)
- [ ] Query rewriting and parent-document retrieval
- [ ] Support for more formats (DOCX, Markdown, HTML)
- [x] Simple web UI

## License

Released under the [MIT License](LICENSE).

## Tech stack

Python · Ollama · PyTorch · sentence-transformers · rank-bm25 · LangChain (text splitters, integrations) · ChromaDB · pypdf
