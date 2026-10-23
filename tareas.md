# Tareas

Lista de trabajo. Marca `[x]` al terminar y anota el resultado. `estado.md` resume dónde estamos; este archivo dice qué falta.

## Preparar el repo para difundirlo

- [x] Corregir la descripción y los topics del repo en GitHub (decía "Phi-3-mini 4-bit", ya no es cierto)
- [x] Script que descargue el PDF del Reglamento General de Circulación desde el BOE, para que cualquiera reproduzca los números (`scripts/download_dgt.py` + README)
- [x] Revisar las preguntas dudosas del eval set DGT (tabla de velocidades art. 48, marcha atrás art. 62, brazo del agente Anexo I, Anexo III 70 km/h)
- [x] Probar la UI con una pregunta real y sacar una captura para el README
- [x] README: dejar claro que el rendimiento con modelos grandes no está medido (solo hay guía y tests con mocks)
- [x] Borrador de post de difusión con los hallazgos (`docs/post-difusion.md`). Publicarlo lo decide el usuario

## Ideas encontradas (investigación 2026-10-23)

- [ ] (en curso, subagente) **Docling** para leer PDFs (docling-project/docling, ~69k estrellas, soporta tablas, DOCX, HTML, PPTX): ataca el error de la tabla de velocidades aplanada y cubre a la vez el ítem de soportar más formatos. Probar con el reglamento y comparar con `pypdf`
- [ ] (en curso, subagente) Probar modelos alternativos con el eval que ya tenemos (cambiar de modelo cuesta una línea de config): embeddings y reranker Qwen3 (0.6B), embeddings Granite multilingües pequeños; LLMs de 2-4B nuevos para 4 GB. Las fuentes web eran poco fiables: decidir solo con nuestras cifras
- [ ] Juez de fidelidad independiente: Ragas (~16k estrellas) o vectara/open-rag-eval (evalúa sin respuestas de referencia), en lugar de que el mismo modelo se juzgue a sí mismo
- [ ] Contextual retrieval (anteponer a cada chunk un contexto generado por el LLM antes de indexar): puede subir el 0.87 de recuperación del libro; cuesta una llamada al LLM por chunk al indexar
- [ ] Publicar en Hugging Face el eval set del reglamento como Dataset (el texto es público, BOE); el del libro no, por derechos de autor
- [ ] Opcional: Space de Hugging Face con la UI de Gradio sobre el reglamento (requiere un LLM pequeño en CPU; evaluar si compensa)

## Mejoras del sistema

- [ ] Medir un modelo grande de verdad (necesita una máquina con VRAM: vLLM u Ollama con 32B-70B y `configs/empresa-multi-gpu.example.yaml`)
- [ ] Query rewriting
- [ ] Chunking que conserve tablas (la tabla de velocidades se aplana y confunde al 7B)
- [ ] Parent-document retrieval
- [ ] Aplicar al prompt inglés `qa.md` el ejemplo de rechazo, solo si se mide mejora
- [ ] Soporte DOCX, Markdown y HTML en `ingest.py`
- [ ] Juez de fidelidad distinto del modelo que responde

## Hecho

- [x] Calibrar `min_rerank_score` (se queda en `null`)
- [x] UI web con Gradio
- [x] Backend LLM intercambiable y perfiles de configuración
- [x] Segundo corpus (Reglamento General de Circulación) con 51 preguntas
- [x] Arreglar el bucle de `citations` y el rechazo del 7B en español
