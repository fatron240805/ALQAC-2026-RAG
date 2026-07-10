# Real Inference Runbook

Use this when moving from dry-run to real ALQAC inference with a local Ollama
model. This pipeline does not require a proprietary LLM API.

## 1. Install and Start Ollama

Install Ollama from https://ollama.com, then make sure the local server is
running at `http://localhost:11434`.

Pull one instruct model under 10B parameters:

```powershell
ollama pull qwen2.5:7b-instruct
```

If that exact tag is not available on your machine, use:

```powershell
ollama pull qwen2.5:7b
ollama list
```

Then copy the exact model name from `ollama list` into `.env`.

## 2. Fill `.env`

`.env` is gitignored. Keep the local Ollama settings and replace only the team
token:

```text
ALQAC_TEAM_TOKEN=<official team token>
ALQAC_API_URL=https://alqac-api.ngrok.pro

ALQAC_LLM_PROVIDER=ollama
ALQAC_LLM_BASE_URL=http://localhost:11434
ALQAC_LLM_MODEL_NAME=qwen2.5:7b-instruct

SECONDS_BETWEEN_API_CALLS=5.0
FAILURE_BACKOFF_SECONDS=2.0
```

For vLLM, llama.cpp server, LM Studio, or another `/v1/chat/completions`
endpoint, set:

```text
ALQAC_LLM_PROVIDER=openai-compatible
```

## 3. Preflight

```powershell
python -m orchestration.run_pipeline check-runtime
python -m orchestration.run_pipeline check-runtime --ping-llm
```

After the official team token is filled, optionally test the case retrieval API:

```powershell
python -m orchestration.run_pipeline check-runtime --ping-llm --ping-case-api
```

`--ping-case-api` uses one official retrieval API call, so do it sparingly.

## 4. Smoke Test

```powershell
python -m orchestration.run_pipeline run --limit 1 --rebuild-index
python -m orchestration.run_pipeline validate-output --require-real
```

## 5. Small Real Batch

```powershell
python -m orchestration.run_pipeline run --limit 5
python -m orchestration.run_pipeline validate-output --require-real
```

## 6. Export Submission Shape

```powershell
python -m orchestration.run_pipeline export-submission --output submission.json
```

Do not submit until local gold labels have been filled and evaluation improves.
