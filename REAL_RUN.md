# Real Inference Runbook

Use this when moving from dry-run to real ALQAC inference with Neo4j-backed
LegalGraphRAG and an API-served open-weight SLM. This pipeline does not require
a proprietary LLM API.

## 1. Start an SLM API

Use one open-weight instruct model under 10B parameters per run.

Option A: Ollama native `/api/chat`. Install Ollama from https://ollama.com,
then make sure the local server is running at `http://localhost:11434`.

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

Option B: OpenAI-compatible `/v1/chat/completions` from vLLM, LM Studio,
llama.cpp server, or a similar self-hosted runtime.

## 2. Fill `.env`

`.env` is gitignored. Replace local secrets and choose exactly one SLM provider:

```text
ALQAC_TEAM_TOKEN=<official team token>
ALQAC_API_URL=https://alqac-api.ngrok.pro

ALQAC_LLM_PROVIDER=ollama
ALQAC_LLM_BASE_URL=http://localhost:11434
ALQAC_LLM_MODEL_NAME=qwen2.5:7b-instruct
# Optional, only for OpenAI-compatible self-hosted endpoints behind auth:
# ALQAC_LLM_API_KEY=<private endpoint key>

NEO4J_URI=neo4j+s://<Aura-instance-id>.databases.neo4j.io
NEO4J_USERNAME=<Aura username>
NEO4J_PASSWORD=<Aura password>
# Leave blank to use the Aura instance default database.
NEO4J_DATABASE=
NEO4J_MAX_CONNECTION_POOL_SIZE=8
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=30
NEO4J_IMPORT_BATCH_SIZE=200

SECONDS_BETWEEN_API_CALLS=5.0
FAILURE_BACKOFF_SECONDS=2.0
```

For vLLM, llama.cpp server, LM Studio, or another `/v1/chat/completions`
endpoint, set:

```text
ALQAC_LLM_PROVIDER=openai-compatible
```

## 3. Build and Import the Legal Graph

Build JSONL graph artifacts first:

```powershell
python -m orchestration.run_pipeline build-graph
```

To use AuraDB Free at runtime, install the optional driver, verify the cloud
connection, and import the graph in small batches:

```powershell
pip install neo4j
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
python -m orchestration.run_pipeline build-graph --import-neo4j --clear-graph --batch-size 200
```

Without a reachable Neo4j server, the pipeline logs a warning and falls back to
flat retrieval.

## 4. Preflight

```powershell
python -m orchestration.run_pipeline check-runtime
python -m orchestration.run_pipeline check-runtime --ping-llm
```

After the official team token is filled, optionally test the case retrieval API:

```powershell
python -m orchestration.run_pipeline check-runtime --ping-llm --ping-case-api
```

`--ping-case-api` uses one official retrieval API call, so do it sparingly.

## 5. Smoke Test

```powershell
python -m orchestration.run_pipeline run --limit 1 --rebuild-index --require-graph
python -m orchestration.run_pipeline validate-output --require-real
```

## 6. Small Real Batch

```powershell
python -m orchestration.run_pipeline run --limit 5 --require-graph
python -m orchestration.run_pipeline validate-output --require-real
```

## 7. Export Submission Shape

```powershell
python -m orchestration.run_pipeline export-submission --output submission.json
```

Do not submit until local gold labels have been filled and evaluation improves.
