# Local RAG System

A fully **local, private Retrieval-Augmented Generation (RAG)** pipeline for asking questions about your own PDF documents. No API keys, no cloud services: documents, embeddings and the language model all stay on your machine.

Answers are grounded in the retrieved text and come with **source citations** (file and page). When the documents don't contain the answer, the model is instructed to say *"I do not know"* instead of inventing one.

## How it works

```
 PDFs ──► ingest ──► chunks + metadata ──► embed ──► ChromaDB (on disk)
                                                          │
 question ──► embed ──► top-k similarity search ◄─────────┘
                              │
                  retrieved chunks + strict prompt
                              │
                   Phi-3-mini (4-bit quantized)
                              │
                     answer + cited sources
```

| Stage | Implementation |
|---|---|
| PDF parsing | `pypdf`, page by page |
| Chunking | `RecursiveCharacterTextSplitter` (512 chars, 50 overlap); each chunk keeps its `source` and `page` |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers`, normalized for cosine similarity |
| Vector store | ChromaDB, persisted locally |
| Generation | `microsoft/Phi-3-mini-4k-instruct`, 4-bit NF4 quantization with `bitsandbytes` |
| Guardrails | Context-only system prompt, greedy decoding, distance-based relevance gate, explicit "I do not know" fallback |

## Project structure

```
├── main.py              # CLI entry point (ingest / query)
├── config.yaml          # Models, chunking, retrieval and generation settings
├── src/
│   ├── ingest.py        # PDF loading and chunking with metadata
│   ├── embed.py         # Embedding model and ChromaDB helpers
│   ├── generator.py     # RAGPipeline: retrieval + local LLM generation
│   └── evaluate.py      # Retrieval evaluation (hit rate, MRR)
└── data/
    ├── raw/             # Put your PDFs here
    └── eval_set.yaml    # Sample evaluation questions
```

## Getting started

```bash
git clone <this-repo> && cd rag-local-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Put one or more PDFs in `data/raw/` (a sample is included).
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

Example output:

```
$ python main.py --mode query --question "What did the narrator draw when he was six years old?"

The narrator drew a picture of a boa constrictor digesting an elephant when he was six years old.

Sources: TheLittlePrince.pdf (page 6), TheLittlePrince.pdf (page 4)
```

Off-topic questions are rejected before reaching the LLM (retrieval distance above `max_distance`):

```
$ python main.py --mode query --question "What is the capital of France?"

I do not know based on the provided documents.
```

### Hardware notes

- 4-bit quantization (`quantization: "4bit"`) requires an **NVIDIA GPU with CUDA**. Without one, the pipeline automatically falls back to unquantized weights, which works on CPU but is slow and needs roughly 8–16 GB of RAM. You can also set `quantization: "none"` explicitly.
- The first run downloads the models from Hugging Face; after that everything works offline.

## Configuration

Everything is controlled from `config.yaml`: model names, `chunk_size`, `chunk_overlap`, `top_k`, `temperature`, `max_new_tokens`, device and paths. To work with non-English documents, switch the embedding model to a multilingual one (e.g. `BAAI/bge-m3`) and re-run ingestion.

## Evaluation

Retrieval quality is measured without needing the LLM, using a small question set with expected keywords (`data/eval_set.yaml`):

```bash
python -m src.evaluate
```

It reports **hit rate@k** (is a relevant chunk in the top-k?) and **MRR** (how high does the first relevant chunk rank?). Add your own questions to the YAML file to benchmark your documents.

Current result on the bundled sample (*The Little Prince*, 240 chunks, `top_k=3`):

| Metric | Score |
|---|---|
| Hit rate@3 | 0.80 |
| MRR | 0.80 |

The one miss is a deliberately broad question ("What does the prince take care of on his planet?"), a typical failure mode of small embedding models on vague queries and a good target for reranking (see roadmap).

## Design decisions

- **Local-first**: suited to confidential documents where sending data to a third-party API isn't an option.
- **Metadata preserved end to end**: chunks are split per page so citations are accurate and chunks never straddle two documents.
- **Models loaded once**: `RAGPipeline` keeps the embedder, vector store and LLM in memory across questions.
- **Idempotent ingestion**: the collection is rebuilt on each ingest.
- **Deterministic by default**: greedy decoding (`temperature: 0`) to favor faithful answers.
- **Relevance gate**: chunks farther than `max_distance` from the question are discarded, and if none remain the system answers "I do not know" without calling the LLM. The threshold was calibrated from observed distances (in-scope questions ≈ 0.67, off-topic ≈ 0.89–1.13).
- **Noise filtering**: near-empty chunks (title pages, page numbers) are dropped at ingestion.

## Known limitations

- **Small-model comprehension.** Phi-3-mini (3.8B) can misattribute who said what when the retrieved context mixes speakers. For example, asked what the prince looks like (a question the book barely answers), it attributed a description of the *snake* to him. A distance threshold can't catch this because the retriever still finds superficially related passages (distance ≈ 0.67, the same as a good match). Mitigations on the roadmap: a cross-encoder reranker and a larger model.
- **Retrieval on vague questions.** Broad questions can miss (see the evaluation above).
- **CPU speed.** Without a CUDA GPU, the unquantized model is slow; each answer takes a while.

## Roadmap

- [ ] Answer-quality evaluation (faithfulness / relevance, e.g. with RAGAS)
- [ ] Reranking with a cross-encoder
- [ ] Support for more formats (DOCX, Markdown, HTML)
- [ ] Simple web UI

## Tech stack

Python · PyTorch · Hugging Face Transformers · bitsandbytes · sentence-transformers · LangChain (text splitters, integrations) · ChromaDB · pypdf
