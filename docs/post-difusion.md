# Borrador de post (para publicar a mano)

> Borrador. Rellena las cifras con la última evaluación antes de publicar. Sitios posibles: r/LocalLLaMA, comunidades hispanas de IA, blog propio.

**Título:** Un RAG 100% local con evaluación medida: lo que aprendí con modelos de 3B y 7B

Construí un RAG que corre entero en mi máquina (GPU de 4 GB): búsqueda híbrida (embeddings + BM25), reranking con cross-encoder, un LLM en Ollama y citas verificadas. No es un producto: es una base pequeña y medible. Lo interesante no son las funciones sino lo que midió.

**Qué medí** (dos textos muy distintos: *El Principito* y el Reglamento General de Circulación de España, 730 páginas):

- Cada etapa de recuperación aporta: en el libro, solo embeddings 0.71 → híbrido 0.79 → con reranker 0.87 de acierto en el top-4. En el manual legal todo ronda 1.00 (MRR 0.94 → 0.97 con reranker), porque cada pregunta tiene términos distintivos.
- Un modelo de 7B no mejoró a uno de 3B en precisión de respuesta (0.76 vs 0.74 en el libro; 0.93 vs 0.88 en el reglamento, una diferencia de dos preguntas). Sí mejoró la fidelidad autoevaluada.
- Rechazar bien lo que no está en los documentos fue lo más fácil de conseguir (1.00), pero hubo que enseñárselo al 7B con un ejemplo.

**Tres fallos que no esperaba**

1. Con salida JSON estructurada, un array de citas sin límite hizo que el modelo entrara en bucle repitiendo números hasta cortar el JSON. Se arregla acotando el esquema a los fragmentos que existen.
2. El 7B, al querer rechazar, escribía la palabra "answerable" dentro del campo de respuesta. Un ejemplo exacto del JSON de rechazo en el prompt lo arregló.
3. Una tabla de velocidades aplanada a texto hizo que el modelo eligiera la cifra de un fragmento vecino. Con el fragmento correcto en el contexto, el error fue de comprensión, no de recuperación.

**Limitaciones honestas**

- Las preguntas de evaluación del reglamento las generó un modelo y se revisaron contra el texto; las palabras clave salen del propio documento, así que el acierto de recuperación es optimista.
- Con 38-42 preguntas, una diferencia de una pregunta es ruido.
- Hay un backend para servidores tipo vLLM pensado para modelos grandes, pero solo está probado con mocks; no tengo medidas con 32B-70B.

Código, evaluación y guía para aprender (en español): https://github.com/BauDahn/rag-local-system
