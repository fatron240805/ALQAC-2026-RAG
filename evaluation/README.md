# GraphRAG Evaluation

`retrieval_benchmark.py` evaluates Neo4j GraphRAG retrieval against
`related_law_provisions` in `data/ALQAC2026_public_test.json`.

## Prerequisites

1. Configure Aura credentials in `.env`.
2. Build the local graph artifacts and import them into Neo4j:

```powershell
python -m orchestration.run_pipeline build-graph --import-neo4j --batch-size 200
```

3. Verify the database connection and graph contents:

```powershell
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
```

## Run the retrieval benchmark

```powershell
python -m evaluation.retrieval_benchmark --retriever graph --top-k 1,3,5,8,10 --output experiments/retrieval_graph_benchmark_report.json --cases-output experiments/retrieval_graph_benchmark_cases.csv
```

The benchmark traverses Neo4j only by default. It compares retrieved legal
provisions with the public test `related_law_provisions` and reports Hit@K,
Precision@K, Recall@K, F1, and MRR. Use `--include-flat-fallback` only when
explicitly measuring the hybrid fallback path.

## Local end-to-end evaluation

For predictions with manually verified labels, run the built-in evaluator:

```powershell
python -m orchestration.run_pipeline evaluate --pred experiments/<prediction-file>.json
```

The local gold file is `data/local_validation_gold.json`. Create or extend it
with `python -m evaluation.scaffold_gold_labels` when needed.
