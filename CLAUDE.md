# Instrucciones para Claude

- Al empezar una sesión, lee `estado.md` para saber en qué punto está el proyecto y cuál es el siguiente paso.
- Al terminar una sesión (o cuando el usuario lo pida), actualiza `estado.md`: fecha, qué se hizo, siguiente paso, decisiones y bloqueos.

## Proyecto

RAG 100% local sobre PDFs (Python, Ollama o servidor compatible con OpenAI, ChromaDB, BM25, reranker). Perfiles de configuración en `config.yaml` (por defecto) y `configs/*.yaml`; usa `--config` en `main.py`. Las rutas relativas del config se resuelven contra la raíz del proyecto. Más detalle en `README.md`, `arquitectura.md` y `docs/`.

## Cómo trabajar aquí

- Tests: `.venv/bin/python -m pytest` (no necesitan modelos). Deben pasar antes de commitear.
- Evaluar: `python main.py --mode eval [--answers] [--config configs/dgt.yaml]`. `--answers` necesita Ollama (`ollama serve`; `sudo systemctl` pide contraseña) y tarda varios minutos con el 7B. El hardware habitual es una GPU de 4 GB: modelos de 3B-7B.
- No commitear PDFs ni índices (`data/raw/*.pdf`, `data/corpora/*/raw/*.pdf`, `vector_db/` están en `.gitignore`).
- Tras cambiar prompts o recuperación, mide con el eval antes de dar algo por mejor; con 38-42 preguntas una diferencia de 1 pregunta es ruido.

## Trampas conocidas

- Las llaves `{}` del prompt de usuario pasan por `.format()`; las del system prompt no. No uses `{{ }}` en el system prompt.
- El orden de las propiedades del esquema JSON importa (`answer`, `citations`, `answerable`). `citations` debe seguir acotado (`answer_schema(n)`), si no el modelo entra en bucle y corta el JSON.
- Con el 7B en español, mostrar el JSON exacto de rechazo en el prompt evita que escriba "answerable" dentro de `answer`.
- En la shell: `pkill -f` / `pgrep -f` pueden coincidir con tu propio comando y matar la shell (exit 144). Mata procesos por PID.
- Los resultados temporales van al scratchpad de la sesión, no al proyecto (los config temporales deben estar en la raíz o en `configs/`).
