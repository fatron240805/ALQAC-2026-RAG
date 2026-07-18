# GraphRAG Runbook

## 1. Configure AuraDB

Copy `env.example` to `.env` and set `NEO4J_URI`, `NEO4J_USERNAME`, and
`NEO4J_PASSWORD`. Leave `NEO4J_DATABASE` empty for the Aura default database
unless the Aura console explicitly provides a database name.

## 2. Build and import the graph

```powershell
python -m orchestration.run_pipeline build-graph --import-neo4j --batch-size 200
```

This generates local graph artifacts under `data/graph/` and imports them into
AuraDB in bounded batches.

## 3. Confirm the live backend

```powershell
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
```

The command must report the `neo4j` backend and a non-zero `LegalNode` count
before running a graph benchmark or production retrieval.

## 4. Benchmark GraphRAG retrieval

```powershell
python -m evaluation.retrieval_benchmark --retriever graph --top-k 1,3,5,8,10 --output experiments/retrieval_graph_benchmark_report.json --cases-output experiments/retrieval_graph_benchmark_cases.csv
```

The gold evidence is read directly from `related_law_provisions` in the public
test. The default is strict Neo4j traversal with no flat-index fallback.

## 5. Run prediction and build a submission

```powershell
python -m orchestration.run_pipeline run --require-graph
python -m orchestration.run_pipeline export-submission --pred experiments/run_graphrag.json --output submission.json
```

Run `python -m orchestration.run_pipeline --help` for all runtime, retrieval,
and model-provider options.
