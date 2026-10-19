# Cómo funciona este proyecto (guía para aprender)

No hace falta saber nada de IA para seguir esta guía. Vamos de lo simple a lo concreto, y al final hay un glosario y ejercicios para probar por tu cuenta. Para el detalle técnico por módulo, mirá [arquitectura.md](arquitectura.md).

## 1. El problema

Tenés un PDF (por ejemplo *El Principito*) y querés preguntarle cosas: "¿qué dibujó el narrador a los seis años?". Un modelo de lenguaje (LLM) como ChatGPT, Claude o Llama sabe escribir muy bien, pero tiene dos problemas:

- **No conoce tu documento.** Si es privado o nuevo, nunca lo vio.
- **Inventa.** Si no sabe, a veces responde algo que suena bien pero es falso.

La solución se llama **RAG** (*Retrieval-Augmented Generation*, generación aumentada con búsqueda). La idea es la de un examen a libro abierto:

1. Cuando llega una pregunta, primero **buscás** en el libro las páginas que parecen relevantes.
2. Le entregás esas páginas al modelo y le decís: "respondé usando solo esto".

Así el modelo no depende de lo que recuerda, sino de lo que tiene delante. Y como sabemos de qué páginas salió, podemos **citar la fuente**.

Este proyecto es "local" porque todo corre en tu computadora: nada se envía a internet.

## 2. Las dos fases

Todo el sistema se resume en dos momentos:

- **Preparar (una vez):** leer los PDFs y organizarlos para poder buscar rápido. Se hace con `python main.py --mode ingest`.
- **Preguntar (cada vez):** buscar, armar el prompt, consultar al modelo y mostrar la respuesta. Se hace con `python main.py --mode query`.

## 3. Fase de preparación, paso a paso

### 3.1 Leer el PDF
`pypdf` extrae el texto de cada página. Guardamos de qué archivo y de qué página viene cada texto.

### 3.2 Partir en trozos (*chunks*)
Un libro entero no cabe en el prompt de un modelo, y además buscar en trozos chicos es más preciso. Entonces el texto se parte en pedacitos de unos 512 caracteres (un par de párrafos), con un poco de solapamiento (50 caracteres) para que una frase cortada en el borde no se pierda. Cada trozo recuerda su archivo y su página. Los trozos casi vacíos (la portada, números de página) se descartan porque solo meten ruido.

### 3.3 Convertir texto en números (*embeddings*)
Esta es la idea central. Un **modelo de embeddings** lee un texto y devuelve una lista de números (aquí, 384) llamada **vector**. Lo importante: textos con significado parecido dan vectores parecidos. "El zorro quiere ser domesticado" y "el animal pide que lo amansen" quedan cerca, aunque casi no compartan palabras. Podés pensarlo como colocar cada trozo en un mapa enorme donde la distancia entre puntos mide qué tan parecido es su significado.

Los vectores se guardan en **ChromaDB**, una base de datos pensada para encontrar "los vecinos más cercanos" de un punto muy rápido. Usamos distancia **coseno**: mide el ángulo entre vectores, no su tamaño.

### 3.4 Un segundo índice, por palabras (BM25)
La búsqueda por significado tiene un punto ciego: los nombres propios y las cifras exactas ("asteroide B-612"). Para eso existe **BM25**, un método clásico que puntúa los trozos según cuántas veces aparecen las palabras de tu pregunta y cuán raras son esas palabras. Es como el Ctrl+F, pero ordenado por relevancia. Se guarda el texto en `chunks.json` y el índice se reconstruye al arrancar.

### 3.5 El manifiesto
`manifest.json` anota con qué modelo y configuración se hizo el índice. Si después cambiás el modelo de embeddings sin reindexar, los vectores guardados ya no "hablan el mismo idioma" que los de tus preguntas, y los resultados serían basura sin ningún error visible. El manifiesto detecta eso y te avisa.

## 4. Fase de pregunta, paso a paso

Seguimos con la pregunta: **"¿Qué creatura muerde al principito al final?"**

### 4.1 Buscar candidatos de dos maneras
- **Por significado:** la pregunta se convierte en vector y Chroma devuelve los 20 trozos más cercanos. A la pregunta se le agrega una frase de instrucción antes de convertirla, porque así lo recomiendan los autores del modelo BGE.
- **Por palabras:** BM25 devuelve los 20 trozos que mejor coinciden con "creatura", "muerde", "principito"...

### 4.2 Fusionar (RRF)
Ahora hay dos listas con puntuaciones que no se pueden comparar (distancias vs. puntajes BM25). La fusión por **rango recíproco** evita el problema: solo mira la *posición* de cada trozo en cada lista y suma `1 / (60 + posición)`. Un trozo que aparece alto en ambas listas gana. Es como pedirle recomendaciones a dos amigos con gustos distintos y quedarte con lo que ambos mencionan.

### 4.3 Reordenar con un reranker
Hasta acá los modelos miraron la pregunta y cada trozo *por separado*. El **reranker** (un *cross-encoder*) lee la pregunta y el trozo *juntos*, y por eso juzga mucho mejor si el trozo realmente responde. Es más lento, así que solo se usa sobre los ~40 candidatos, no sobre todo el libro. Nos quedamos con los 4 mejores.

Esta combinación (buscar rápido y amplio, después ordenar fino) es el patrón estándar en sistemas de búsqueda reales.

### 4.4 Armar el prompt
Un **prompt** es el texto que se le entrega al LLM. El nuestro está en `src/rag/prompts/qa.md` y tiene dos partes:

- **Instrucciones del sistema:** "usá solo los fragmentos; si no alcanzan, decí que no se puede responder; citá los números de los fragmentos que usaste".
- **Mensaje del usuario:** los 4 trozos numerados (`[1] TheLittlePrince.pdf, p.58 ...`) y tu pregunta.

Numerar los trozos es lo que permite al modelo decir "esto lo saqué de [2]".

### 4.5 Consultar al LLM con Ollama
**Ollama** es un programa que ejecuta modelos de lenguaje en tu máquina y los ofrece mediante una API local. Nuestro código le manda el prompt y espera la respuesta. Ventaja: para probar otro modelo basta con cambiar su nombre en `config.yaml`.

Le pedimos que responda en **JSON** con una forma fija:

```json
{ "answerable": true, "answer": "Una serpiente.", "citations": [2] }
```

Con ese formato obligatorio no hay que "adivinar" leyendo texto libre si el modelo se rindió o qué fuentes usó.

### 4.6 Verificar antes de mostrar
Los modelos pequeños a veces se equivocan, así que el código comprueba:

- Si `answerable` es falso, se muestra "I do not know based on the provided documents." y ninguna fuente.
- Las citas se comparan con los fragmentos que realmente se le dieron. Si el modelo cita `[9]` y solo había 4, esa cita se descarta.
- Si respondió pero no citó nada válido, el programa avisa que la respuesta no se pudo rastrear a una fuente.

## 5. Mapa del código

| Archivo | Qué hace, en una frase |
|---|---|
| `main.py` | La puerta de entrada: lee lo que escribís en la terminal. |
| `config.yaml` | El panel de control: modelos, tamaños, textos. |
| `src/rag/ingest.py` | Lee PDFs y los parte en trozos. |
| `src/rag/index.py` | Guarda los trozos en Chroma, BM25 y escribe el manifiesto. |
| `src/rag/retrieve.py` | Busca, fusiona y reordena. |
| `src/rag/llm.py` | Habla con Ollama. |
| `src/rag/pipeline.py` | Une todo para responder una pregunta. |
| `src/rag/prompts/qa.md` | El texto de instrucciones para el modelo. |
| `src/rag/evaluate.py` | Mide si el sistema funciona bien. |
| `tests/` | Pruebas automáticas que protegen el código al cambiarlo. |

Para seguir una pregunta por el código, empezá en `RAGPipeline.ask` (`src/rag/pipeline.py`): es corto y cada línea llama a una de las etapas de la sección 4.

## 6. ¿Cómo sabemos si funciona? Evaluación

Ver que "una respuesta se ve bien" no alcanza. Por eso `data/eval_set.yaml` tiene preguntas con palabras clave esperadas, y `python main.py --mode eval` mide dos números:

- **Hit rate@k:** de las preguntas, ¿en cuántas aparece un trozo relevante entre los k primeros?
- **MRR:** en promedio, ¿qué tan arriba queda el primer trozo relevante? (1 si es el primero, 0,5 si es el segundo, 0,33 si el tercero...)

Se prueban tres variantes (solo significado, híbrido, híbrido + reranker) para ver con datos si cada pieza aporta. Con `--answers` también se verifica que el sistema rechace preguntas imposibles ("¿capital de Francia?") y que un segundo modelo, usado como juez, confirme que las respuestas están respaldadas por los trozos citados.

## 7. Glosario

| Término | Significado simple |
|---|---|
| **LLM** | Modelo de lenguaje: el programa que escribe la respuesta. |
| **RAG** | Buscar primero, responder después con lo encontrado. |
| **Chunk** | Un trozo del documento, de unos pocos párrafos. |
| **Embedding / vector** | Lista de números que representa el significado de un texto. |
| **Distancia coseno** | Medida de qué tan parecidos son dos vectores. |
| **Base vectorial** | Base de datos que encuentra rápido los vectores más cercanos (ChromaDB). |
| **BM25** | Búsqueda clásica por palabras clave con puntuación. |
| **RRF** | Truco para combinar dos rankings usando solo posiciones. |
| **Reranker** | Modelo que reordena candidatos leyendo pregunta y trozo juntos. |
| **Prompt** | El texto completo que se le manda al LLM. |
| **Cuantización** | Comprimir un modelo para que ocupe menos memoria (los modelos de Ollama ya vienen así). |
| **VRAM** | Memoria de la tarjeta gráfica; limita el tamaño del modelo que podés usar. |
| **Manifiesto** | Archivo que anota con qué configuración se hizo el índice. |
| **Alucinación** | Cuando el modelo inventa algo que suena creíble. |

## 8. Ejercicios para aprender haciendo

1. **Mirá un chunk.** Abrí `data/vector_db/chunks.json` y leé cuatro o cinco trozos. ¿Están bien cortados? ¿Qué pasaría con `chunk_size: 200` o `1500`? Cambialo, reingestá y corré `--mode eval` para ver el efecto.
2. **Compará modos.** Corré `python main.py --mode eval` y mirá en qué pregunta falla cada modo. ¿Por qué el reranker acierta donde los otros no?
3. **Rompé el prompt a propósito.** Editá `src/rag/prompts/qa.md` y quitá la regla de "usar solo los fragmentos". Preguntá algo que el libro no diga y mirá si el modelo inventa.
4. **Probá otro modelo.** `ollama pull llama3.2:3b`, cambiá `models.llm` y repetí las mismas preguntas.
5. **Provocá el manifiesto.** Cambiá `models.embedding` sin reingestar y ejecutá una consulta: verás el aviso que protege el índice.
6. **Agregá tu propio PDF** a `data/raw/`, reingestá y escribí 5 preguntas en `eval_set.yaml` sobre él.
7. **Seguí una pregunta con el debugger** o con `print` dentro de `Retriever.retrieve` para ver qué trozos pasan por cada etapa.
