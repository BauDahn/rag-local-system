# Estado del proyecto

> **Para Claude:** lee este archivo al empezar cada sesión. Al terminar, actualízalo (fecha, "Hecho", "Siguiente paso", "Decisiones", "Bloqueos") antes de cerrar o commitear.

**Última actualización:** 2026-10-23 (segundo corpus DGT, backend para modelos grandes, bugs de JSON)

## Qué es

Sistema RAG 100% local sobre PDFs: ingesta → híbrido (dense + BM25, RRF) → rerank cross-encoder → LLM en Ollama con salida JSON `{answer, answerable, citations}` → citas verificadas. Detalles en `README.md`, `arquitectura.md` y `COMO_FUNCIONA.md`.

## Dónde estamos

- Pipeline completo y funcionando: ingesta, recuperación híbrida, reranking, generación, verificación de citas, manifest del índice.
- Evaluación: 46 preguntas (38 respondibles, 8 no) en `data/eval_set.yaml`, sobre *El Principito* (el PDF no está en el repo; `data/raw/*.pdf` está en `.gitignore`).
- Resultados medidos (`top_k=4`): hit rate@4 0.87 y MRR 0.76 con hybrid+rerank; precisión de respuesta 0.74 (3B) / 0.76 (7B); rechazo correcto 1.00; fidelidad 0.54 (3B) / 0.79 (7B).
- Modelo por defecto: `qwen2.5:3b-instruct` (cabe en ~4 GB de VRAM). El 7B no mejora la precisión, así que el cuello de botella es el prompt y la evaluación, no el tamaño.
- Segundo corpus DGT (español) indexado en `data/corpora/dgt/vector_db` (git-ignorado). Perfiles en `configs/`.
- Tests: `pytest` (41, sin modelos). Rama `main`, árbol limpio al cierre de la sesión (ver `git log`).

## Siguiente paso (prioridad)

1. **Medir un modelo grande de verdad** (el objetivo del proyecto: ~90% con 2 GPUs). Hace falta una máquina con VRAM: levantar vLLM o Ollama con un 32B-70B y correr `--config configs/empresa-multi-gpu.example.yaml --mode eval --answers` sobre ambos corpus. Hoy el backend `openai` solo está probado con mocks.
2. **Probar la UI con una pregunta real** (`python main.py --mode ui`; Ollama con `ollama serve`, `sudo systemctl` pide contraseña).
3. **Revisar el eval set DGT a mano**: lo generó un subagente y las keywords salen del propio texto, así que el 1.00 de retrieval es optimista. Preguntas dudosas: tabla de velocidades (art. 48), orden de marcha atrás (art. 62), Anexo III 70 km/h.
4. Query rewriting; parent-document retrieval; chunking que conserve tablas (la tabla de velocidades se aplana a texto y confunde al 7B).
5. Aplicar al prompt inglés `qa.md` el ejemplo de rechazo si el 3B/7B escribe "answerable" en `answer`.

## Hecho en la última sesión

- Calibrado `min_rerank_score` (se queda en `null`).
- UI Gradio (`src/rag/ui.py`, `--mode ui`).
- Backend LLM intercambiable: `llm_backend: ollama | openai` (vLLM/llama.cpp), perfiles con `--config`, `configs/dgt.yaml`, `configs/empresa-multi-gpu.example.yaml`, `docs/despliegue-modelos-grandes.md`. Rutas relativas ahora se resuelven contra la raíz del proyecto.
- Segundo corpus: Reglamento General de Circulación (`data/corpora/dgt/raw/rgc.pdf`, git-ignorado) con 51 preguntas en español. Resultados en el README.
- Bug: `citations` sin límite hacía que el modelo entrara en bucle de números y cortara el JSON (esto explicaba los errores de la variante v2 del prompt). Ahora `answer_schema(n)` lo limita a los fragmentos existentes.
- Bug: el 7B escribía "answerable" dentro de `answer` al querer rechazar; `qa_es.md` ahora muestra el JSON de rechazo exacto.
- `evaluate`: no aborta con errores del LLM, normaliza saltos de línea al comparar, y corregí 3 `expected` discutibles del eval de El Principito (eco, zorro, estrellas).
- Resultados DGT (51 preguntas): retrieval hit@4 1.00 / MRR 0.94 (dense), 0.94 (hybrid), 0.97 (rerank); respuestas 3B / 7B: rechazo 1.00 / 1.00, precisión 0.88 / 0.93, fidelidad 0.38 / 0.86 (medido tras corregir 5 preguntas del eval).
- Experimentos de prompt inglés (3B): v2 descartada (0.37, era el bug de citas), v3 0.76 vs 0.74 base (ruido), no adoptada.

## Backlog

- Soporte para DOCX, Markdown y HTML en `ingest.py`.
- Eval set con preguntas sobre otros documentos (hoy solo un libro; 46 preguntas no separan modelos que difieren pocos puntos).
- Fidelidad evaluada por el mismo modelo que responde: probar un juez distinto.

## Decisiones tomadas (no reabrir sin motivo)

- `min_rerank_score` se queda en `null` (calibrado 2026-10-23). Score máximo del top-4 (sigmoide 0-1): las 8 no respondibles dan 0.00-0.09; las 38 respondibles van de 0.00 a 0.98 y 7 quedan por debajo de 0.05 aun teniendo el chunk correcto. Un umbral de 0.05 frenaría 7/8 no respondibles pero rechazaría ~18% de las respondibles. Solo reabrir si cambia el reranker o el idioma/documentos.
- Cambios al prompt `qa.md` no mueven la métrica más allá del ruido (ver arriba); no insistir sin mejorar antes el eval set. Cuidado: reglas que piden reformular ("con tus palabras") hacen que el 3B genere JSON inválido.
- Ollama como servicio LLM intercambiable (`models.llm` o variable `RAG_LLM`).
- En el esquema JSON, `answer` va antes que `answerable` (forzar la decisión primero disparaba los rechazos).
- `temperature: 0`; chunks de 512 con solape 50, partidos por página para citas exactas.
- No se incluyen PDFs con copyright en el repo.

## Comandos útiles

```bash
source .venv/bin/activate
pytest
python main.py --mode ingest
python main.py --mode query --question "..."
python main.py --mode eval                # solo retrieval
python main.py --mode eval --answers      # necesita Ollama
RAG_LLM=qwen2.5:7b-instruct python main.py --mode eval --answers
```

## Bloqueos / notas abiertas

- Ninguno por ahora.
