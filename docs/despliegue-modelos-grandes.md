# Despliegue con modelos grandes (varias GPUs)

El RAG no cambia para usar un modelo de 32B-70B: solo cambia la configuración. El LLM se sirve aparte (vLLM, llama.cpp server...) y el RAG le habla por HTTP con la API de OpenAI. Perfil de partida: `configs/empresa-multi-gpu.example.yaml`.

## 1. Levantar vLLM con 2 GPUs

En el servidor (instalación y requisitos de vLLM: ver su documentación oficial):

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 --port 8000
# opcional: --api-key MI_CLAVE
```

- `--tensor-parallel-size 2` reparte el modelo entre las dos GPUs.
- `--max-model-len` es la ventana de contexto máxima; más contexto consume más VRAM (caché KV).
- `--gpu-memory-utilization` deja margen para otros procesos. Si el embedder y el reranker del RAG corren en las mismas GPUs (`*_device: cuda`), bájalo (p. ej. 0.80) para que quepan.
- `--host 127.0.0.1` mantiene el servicio solo en local; si el RAG corre en otra máquina, protege el puerto con `--api-key` y red interna.
- Comprueba que responde: `curl http://127.0.0.1:8000/v1/models`.

## 2. Apuntar el RAG al servidor

```bash
cp configs/empresa-multi-gpu.example.yaml configs/empresa.yaml
```

Ajusta en `configs/empresa.yaml`:

- `models.llm_backend: openai`
- `models.llm`: exactamente el nombre que devuelve `/v1/models` (el de `vllm serve`, o `--served-model-name`).
- `llm_server.base_url`, `llm_server.timeout` y, si hay clave, `llm_server.api_key` o la variable `RAG_LLM_API_KEY` (tiene prioridad sobre el archivo).
- `paths.*`: carpetas de documentos e índice de este perfil.

Después:

```bash
python main.py --config configs/empresa.yaml --mode ingest
python main.py --config configs/empresa.yaml --mode query --question "..."
python main.py --config configs/empresa.yaml --mode eval --answers
python main.py --config configs/empresa.yaml --mode ui
```

Si cambias de embedder o de tamaño de chunk hay que reingestar (el manifiesto lo avisa). Usa una carpeta `vector_store` distinta por perfil.

Otros servidores compatibles (llama.cpp `llama-server`, SGLang, LM Studio) funcionan igual con `base_url` y `llm` ajustados. Con Ollama en una máquina grande basta `llm_backend: ollama` y, en la sección `ollama`, `num_ctx` y `keep_alive`.

## 3. Elegir modelo según la VRAM

Orden de magnitud para pesos del modelo: unos 2 bytes por parámetro en 16 bits, ~1 byte en 8 bits y ~0,5 bytes en 4 bits. A eso hay que sumar la caché KV (crece con el contexto y con las peticiones simultáneas) y el espacio del embedder y reranker si comparten GPU.

| Modelo | Pesos aprox. en 16 bits | Pesos aprox. en 4 bits |
|---|---|---|
| 7-8B | ~16 GB | ~4-5 GB |
| 32B | ~64 GB | ~16-20 GB |
| 70B | ~140 GB | ~35-40 GB |

Lectura práctica con 2 GPUs (suma de VRAM `V`):

- Un 32B en 16 bits necesita algo más de 64 GB repartidos entre ambas; con 2x48 GB hay margen cómodo, con 2x24 GB conviene una versión cuantizada (AWQ/GPTQ, a 8 o 4 bits).
- Un 70B en 16 bits solo cabe con ~160 GB o más en total; en 4 bits cabe en unos 48 GB totales, con poco margen para contexto largo.
- Son estimaciones de memoria, no de calidad: la calidad con tu corpus se mide con `--mode eval --answers` sobre tu propio `eval_set.yaml`, comparando modelos con el mismo conjunto de preguntas.

## 4. Qué subir con modelos grandes

- `num_ctx` (Ollama) o `--max-model-len` (vLLM): sube a 8k-16k o más si la VRAM lo permite; con `top_k` alto y chunks de 800 caracteres el prompt crece.
- `top_k`: de 4 a 6-10. Un modelo grande ignora mejor el ruido y aprovecha más evidencia.
- `k_candidates`: de 20 a 30-50, para que el reranker tenga más donde elegir (el reranker en `cuda` lo hace barato).
- `max_new_tokens` y `llm_server.timeout`: más holgados, porque los modelos grandes responden más lento y con más detalle.
- `embedding_device` y `reranker_device: cuda` si queda VRAM libre.
- Cambia un parámetro cada vez y compara con `--mode eval` para ver si realmente ayuda.
