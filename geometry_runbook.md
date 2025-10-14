# Geometry Inference Runbook

This guide covers only the geometry solver. Follow the steps below to run tasks with either local Ollama models or popular cloud LVLM services.

## 1. Prerequisites
- Activate the project virtual environment: `source .venv/bin/activate`
- Dependencies for geometry (already installed in the project): `sympy`, `ipykernel`, `ollama` (if you plan to use local models)
- Geometry solver entry point: `python -m geometry.solver`
- Optional evaluation helper: `scripts/analyze_geometry.py`

## 2. Local LVLM via Ollama
1. Ensure the Ollama daemon is running: `ollama serve`
2. Pull a vision-capable model, e.g. `ollama pull llama3.2-vision:11b`
3. Run a single task (defaults inside `geometry/solver.py`):
   ```bash
   MODEL_API_TYPE=ollama \
   MODEL_NAME=llama3.2-vision:11b \
   OLLAMA_STREAM=false \
   python -m geometry.solver
   ```
4. Batch over a dataset (edit the glob as needed):
   ```bash
   MODEL_API_TYPE=ollama MODEL_NAME=llama3.2-vision:11b OLLAMA_STREAM=false \
   python - <<'PY'
   from pathlib import Path
   from geometry.solver import run_geo_task

   base = Path("dataset/geometry/Dataset_GeomVerse")
   out_root = Path("outputs/geometry/llama3.2-vision")
   for task_dir in sorted(p for p in base.iterdir() if p.is_dir()):
       run_geo_task(str(task_dir), str(out_root / task_dir.name), task_type="visuothink", verbose=False)
   PY
   ```
5. Review outputs (one folder per task under `outputs/geometry/...`).
6. Summarise accuracy: `scripts/analyze_geometry.py outputs/geometry/llama3.2-vision`

## 3. Cloud LVLM Options
Set the environment so `geometry/utils_llm.py` routes through Autogen's OpenAI-compatible client. Replace placeholders with your credentials.

### 3.1 OpenAI (GPT-4o family)
```bash
export MODEL_API_TYPE=openai
export MODEL_NAME=gpt-4o
export OPENAI_API_KEY=sk-...
python -m geometry.solver
```

### 3.2 Anthropic (Claude 3.5 Sonnet)
```bash
export MODEL_API_TYPE=anthropic
export MODEL_NAME=claude-3-5-sonnet-20240620
export ANTHROPIC_API_KEY=key-...
python -m geometry.solver
```

### 3.3 Google Vertex / Gemini 1.5 Pro
```bash
export MODEL_API_TYPE=google
export MODEL_NAME=gemini-1.5-pro
export GOOGLE_API_KEY=...  # or set up ADC for service accounts
python -m geometry.solver
```

### 3.4 Groq (Mixtral / LLaMA Vision hosted)
```bash
export MODEL_API_TYPE=groq
export MODEL_NAME=llama-3.2-90b-vision-preview
export GROQ_API_KEY=...
python -m geometry.solver
```

> Tip: keep `MODEL_API_TYPE` aligned with the provider name expected by Autogen (see `autogen/oai/client.py` for exact strings). All other environment variables (`MAX_REPLY`, etc.) can be set the same way as for navigation tasks.

## 4. Hugging Face модели (локальные/Hub)
1. Убедитесь, что установлены зависимости: `pip install -r requirements.txt` (включая `transformers`, `accelerate`, `sentencepiece`, `bitsandbytes`). Для Flash Attention потребуется отдельная установка: `pip install flash-attn --no-build-isolation` (только для CUDA-GPU).
2. Укажите тип API и модель:
   ```bash
   MODEL_API_TYPE=hf \
   MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   HF_DEVICE=cpu \
   HF_MAX_NEW_TOKENS=256 \
   python -m geometry.solver
   ```
   Используйте `HF_MODEL_PATH`, если модель скачана локально, а также параметры `HF_DO_SAMPLE`, `HF_TOP_P`, `HF_TOP_K`, `HF_LOAD_IN_4BIT` и др. для тонкой настройки. Если модель не поддерживает изображения, в подсказку автоматически попадёт текстовая подпись `[Attached image(s): ...]`.
3. После запуска в каждой папке задачи появится файл `metrics.json`, а агрегированная история сохранится под `outputs/geometry/stats/history.jsonl`. Благодаря этому можно вести журнал качества по моделям Hugging Face и Ollama.
4. Для QLoRA/Flash-Attn режима на больших моделях установите GPU-зависимости и задайте:
   ```bash
   MODEL_API_TYPE=hf \
   MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct \
   HF_DEVICE=cuda \
   HF_LOAD_IN_4BIT=true \
   HF_4BIT_COMPUTE_DTYPE=bfloat16 \
   HF_ATTENTION_IMPL=flash_attention_2 \
   MAX_REPLY=12 \
   python -m geometry.solver
   ```
   Параметры `HF_4BIT_QUANT_TYPE`, `HF_4BIT_USE_DOUBLE_QUANT`, `HF_4BIT_COMPUTE_DTYPE` и др. можно настраивать через переменные окружения; Flash Attention автоматически откатится к SDPA, если библиотека `flash-attn` не доступна.

## 5. Post-run Evaluation
- Inspect `output.json` and `output.log` inside each task folder.
- Run `scripts/analyze_geometry.py <outputs_dir>` to get success rate, average turns, and whether the final numeric answer matches the ground truth (within 1e-2 tolerance).
- Example: `scripts/analyze_geometry.py outputs/geometry/llama3.2-vision`

## 6. Common Adjustments
- To change the task set, edit the paths passed to `run_geo_task`.
- Increase `MAX_REPLY` (export env var) if the model needs more steps.
- For cloud runs, ensure your account has image-enabled endpoints; otherwise remove `<img>` tags by switching `_supports_image_messages` logic if necessary.
