# Arquitectura

Este documento describe cómo está construido el sistema: módulos, flujo de datos, decisiones de diseño y cómo extenderlo. Si buscás una explicación para aprender desde cero, leé primero [COMO_FUNCIONA.md](COMO_FUNCIONA.md).

## Vista general

El sistema tiene dos flujos independientes que comparten el índice en disco:

```
INGESTIÓN (se ejecuta una vez por conjunto de documentos)

 data/raw/*.pdf ─► ingest.py ─► chunks + metadatos ─► index.py ─┬─► Chroma (vectores, coseno)
                                                                ├─► chunks.json (texto para BM25)
                                                                └─► manifest.json (versión del índice)

CONSULTA (se ejecuta por cada pregunta)

 pregunta ─► retrieve.py ─► pipeline.py ─► llm.py (Ollama u OpenAI-compat) ─► pipeline.py ─► respuesta + fuentes
              │                │                                  │
              │  denso top-20  │  arma contexto numerado          │  valida citas
              │  BM25  top-20  │  + plantilla de prompt           │  decide "no sé"
              │  RRF + rerank  │
```

## Módulos (`src/rag/`)

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `config.py` | Carga un perfil (`config.yaml` o `--config`) y convierte las rutas a absolutas desde la raíz del repo. | — |
| `ingest.py` | Lee PDFs página por página, los parte en chunks, descarta los casi vacíos y calcula el hash de los PDFs. | pypdf, LangChain splitters |
| `index.py` | Construye y abre el índice: Chroma (coseno), BM25 y `manifest.json`. Valida que el índice coincida con la config. | ingest, Chroma, rank-bm25 |
| `retrieve.py` | Busca candidatos (denso + BM25), los fusiona con RRF y los reordena con el reranker. Expone tres modos: `dense`, `hybrid`, `hybrid_rerank`. | index, sentence-transformers |
| `llm.py` | Backends del LLM (`OllamaLLM`, `OpenAICompatLLM`) con la fábrica `create_llm`. Piden la respuesta como JSON con un esquema y traducen los errores a `LLMError`. | ollama, httpx |
| `pipeline.py` | Orquesta una pregunta: recuperar, armar el prompt, llamar al LLM, validar citas, devolver `RAGAnswer`. | retrieve, llm, index |
| `evaluate.py` | Mide retrieval (hit rate, MRR) por modo y, opcionalmente, rechazos y fidelidad de las respuestas. | retrieve, pipeline |
| `prompts/qa.md` | Plantilla del prompt (secciones `---system---` y `---user---`). | — |

`main.py` es solo la CLI: interpreta argumentos y llama a los módulos anteriores.

## Datos que se guardan en disco (`data/vector_db/`)

| Archivo | Contenido | Para qué |
|---|---|---|
| `chroma.sqlite3` y carpeta UUID | Vectores de los chunks y sus metadatos | Búsqueda por significado |
| `chunks.json` | Texto y metadatos de cada chunk | Reconstruir el índice BM25 al arrancar y recuperar el chunk por `chunk_id` |
| `manifest.json` | Modelo de embeddings, tamaño de chunk, solapamiento, filtro mínimo, hash de PDFs, nº de chunks | Detectar que la config cambió desde que se indexó |

Cada chunk lleva `source` (archivo), `page` y `chunk_id` (su posición). El `chunk_id` es la clave común que permite unir los resultados de Chroma, de BM25 y del reranker.

## Flujo de una consulta, paso a paso

1. **Candidatos densos.** La pregunta se convierte en un vector (con el prefijo de instrucción de BGE) y Chroma devuelve los 20 chunks más cercanos.
2. **Candidatos por palabras.** BM25 puntúa todos los chunks por coincidencia de términos y devuelve los 20 mejores.
3. **Fusión (RRF).** Cada chunk suma `1 / (60 + posición)` por cada lista en la que aparece. Los que quedan bien en ambas listas suben.
4. **Reranking.** Un cross-encoder lee pregunta y chunk juntos y le da una puntuación más precisa. Se queda con los 4 mejores (`top_k`).
5. **Prompt.** Los chunks se numeran como `[1] archivo, p.X` y se insertan en la plantilla junto con la pregunta.
6. **Generación.** Ollama responde con un JSON `{answerable, answer, citations}`, forzado por un esquema.
7. **Validación.** Si `answerable` es falso o la respuesta está vacía, se devuelve el texto de rechazo configurado y ninguna fuente. Si no, se conservan solo las citas que apuntan a fragmentos realmente entregados; si no queda ninguna, la respuesta se marca como `unverified`.

## Decisiones de diseño y compromisos

| Decisión | Por qué | Costo |
|---|---|---|
| **Ollama como servicio** en lugar de cargar el modelo en el proceso | Cambiar de modelo es editar una línea; el código no depende de la versión de `transformers`; el arranque es rápido | Hay que instalar y tener corriendo un programa aparte |
| **Híbrido (denso + BM25)** | El denso entiende el significado pero falla con nombres o cifras exactas; BM25 es lo contrario | Un índice más que mantener |
| **RRF** para fusionar | Solo usa posiciones, no hace falta comparar puntuaciones de escalas distintas | Ignora cuánto mejor es un resultado que otro |
| **Reranker cross-encoder** | Mucho más preciso que comparar vectores, pero lento; por eso solo se aplica a ~40 candidatos | Añade latencia y un modelo extra (en CPU por defecto) |
| **Salida JSON con esquema** | Abstención y citas no dependen de interpretar texto libre | Modelos muy pequeños pueden seguir peor el esquema |
| **Citas validadas** | Las fuentes mostradas son las que el modelo dice haber usado y existen en el contexto | Que cite un fragmento no garantiza que lo haya entendido bien; por eso existe el modo de evaluación de fidelidad |
| **Manifiesto del índice** | Un índice hecho con otro modelo de embeddings devuelve basura sin avisar | Hay que reingestar tras cambiar esos parámetros |
| **Reingestión total** | Simple y sin duplicados | No es incremental: con muchos documentos tarda más |
| **Chunking por página** | Las citas son exactas y un chunk nunca mezcla documentos | Una idea que cruza páginas queda partida |

## Configuración (`config.yaml` y perfiles en `configs/`)

`config.yaml` es el perfil por defecto; cualquier otro se elige con `--config RUTA` en todos los modos (`ingest`, `query`, `eval`, `ui`). Las rutas relativas de `paths` se resuelven contra la raíz del proyecto, no contra la carpeta del perfil. Perfiles incluidos: `configs/dgt.yaml` (corpus en español) y `configs/empresa-multi-gpu.example.yaml` (vLLM con varias GPUs). Ver [docs/despliegue-modelos-grandes.md](docs/despliegue-modelos-grandes.md).

| Sección | Claves | Efecto |
|---|---|---|
| `models` | `embedding`, `reranker`, `llm`, `llm_backend`, `*_device` | Qué modelos se usan y dónde corren. `llm_backend` es `ollama` (por defecto) u `openai` (servidor compatible, p. ej. vLLM); `llm` es la etiqueta de Ollama o el nombre del modelo servido. |
| `ollama` | `host`, `timeout`, `num_ctx`, `keep_alive` | Servidor de Ollama, ventana de contexto y tiempo en memoria. |
| `llm_server` | `base_url`, `api_key` (o `RAG_LLM_API_KEY`), `timeout` | Servidor compatible con OpenAI (solo con `llm_backend: openai`). |
| `parameters` | `chunk_size`, `chunk_overlap`, `min_letters` | Cómo se trocea. Cambiarlos exige reingestar (el manifiesto lo detecta). |
| `parameters` | `k_candidates`, `top_k`, `rrf_k`, `min_rerank_score` | Cuántos candidatos se buscan, cuántos llegan al LLM y el umbral opcional del reranker. |
| `parameters` | `temperature`, `max_new_tokens` | Generación. Con `0` la respuesta es determinista. |
| `prompts` | `template`, `fallback` | Qué plantilla de `prompts/` se usa y el texto de rechazo. |
| `paths` | `raw_data`, `vector_store`, `eval_set` | Carpetas de entrada y de índice, y preguntas de evaluación (por defecto `data/eval_set.yaml`). |

## Cómo extender el sistema

- **Cambiar de modelo:** `ollama pull <modelo>` y editar `models.llm`. Para modelos grandes en varias GPUs, `llm_backend: openai` (ver `docs/despliegue-modelos-grandes.md`); `llm.py` expone `create_llm(config)` y ambos backends comparten `chat_json(messages, schema)`.
- **Cambiar el prompt:** editar `src/rag/prompts/qa.md` o crear otro archivo y apuntar `prompts.template` a él. Los marcadores `{context}` y `{question}` son obligatorios en la sección de usuario. Si cambiás el formato de salida, hay que actualizar también `ANSWER_SCHEMA` en `llm.py` y la lectura en `pipeline.py`.
- **Documentos en otro idioma:** usar un embedding multilingüe (por ejemplo `BAAI/bge-m3`) y un reranker multilingüe, y reingestar.
- **Nuevo formato de documento:** agregar un lector en `ingest.py` que devuelva `Document` con `source` y `page`; el resto no cambia.
- **Nueva etapa de retrieval** (por ejemplo reescritura de consulta): añadirla en `Retriever.retrieve` y un modo nuevo en `MODES` para poder compararla en `evaluate.py`.

## Pruebas

- `tests/` usa un retriever y un LLM falsos, así que corre en segundos y sin descargar modelos. Cubre fusión RRF, validación de citas, construcción del contexto, lectura de prompts, manifiesto del índice y coincidencia de keywords.
- `python main.py --mode eval` mide la calidad de recuperación con modelos reales; `--answers` añade rechazos y fidelidad (requiere Ollama).

## Limitaciones conocidas

- El índice se reconstruye completo en cada ingestión.
- La evaluación usa un solo libro y pocas preguntas, así que las métricas son orientativas.
- Con modelos de ~3B, la comprensión del contexto es limitada; un modelo mayor mejora la calidad si el hardware lo permite.
- No hay filtro de relevancia previo al LLM hasta calibrar `min_rerank_score`; el rechazo de preguntas fuera de tema lo decide el modelo.
