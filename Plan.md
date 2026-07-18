**Core Strategy**

Build an **open-weight LegalGraphRAG baseline** for ALQAC:

- Model: open-weight, `<10B`
- No ChatGPT / Claude / Gemini in the competition system
- Retrieval must be graph-aware, not just flat similarity search
- Reasoning must be evidence-verified before final prediction
- Optimize through local validation, ablations, and disciplined submissions

Reference direction from LegalGraphRAG:

- **Hierarchical Legal Graph**: organize legal sources into layers such as ontology, fact, and rule
- **Neo4j Graph Store**: persist graph nodes, edges, provenance, and traversal state in Neo4j
- **Researcher**: retrieve candidate evidence from the graph and source corpus
- **Auditor**: verify whether evidence is actually supported by the source text
- **Adjudicator**: synthesize verified evidence into the final 4-label answer

**Recommended Architecture**

```text
Question
  ->
Query Analyzer
  - detect legal issue type
  - extract entities, dates, statutes, claims
  - decide graph layer priority
  ->
Graph Retriever
  - normalize legal issue, parties, claims, dates, cited statutes
  - run hybrid lexical + dense seed retrieval over article/clause nodes
  - map seed chunks to graph nodes
  - query Neo4j for node neighborhoods and legal paths
  - route expansion across ontology, fact, and rule layers
  - traverse article -> clause -> point -> concept -> related rule neighbors
  - build source-backed evidence chains
  - fuse lexical, dense, graph, citation, and freshness scores
  ->
Researcher Agent
  - collect candidate evidence chains
  - rank by relevance and coverage
  ->
Auditor Agent
  - verify evidence against source documents
  - reject unsupported or weak passages
  - keep only grounded evidence
  ->
Adjudicator Agent
  - apply facts to rules
  - choose exactly one label
  - produce concise justification
  ->
Verifier
  - check label, evidence, and JSON consistency
  - retry or fallback on low confidence
  ->
Submission Output
```

**Current Baseline Status**

The repo already has a usable offline baseline:

- corpus preparation
- legal corpus cleaning and multi-granular chunking
- deprecated-law filtering
- hybrid BM25 + dense retrieval
- query routing and reranking
- citation usefulness filtering
- orchestration and evaluation CLI

This is a strong starting point, but it is still a flat RAG stack. The next step is to reorganize it into a LegalGraphRAG retrieval pipeline:

- build a hierarchical legal graph from the corpus
- persist the graph in Neo4j as the primary graph store
- keep hybrid retrieval as the seed retriever, not the final retriever
- route questions by graph layer and legal issue type
- expand from seed chunks to graph neighbors before reranking
- add evidence verification before reasoning
- make retrieval return evidence chains, not isolated chunks

**Multi-Granular Chunking**

Legal text is dense. Pattern-based chunking using legal section markers is superior to semantic or sentence-level chunking for preserving legal context.

Chunking policy:

- first split by legal markers such as chapter, article, clause, item, and numbered headings
- keep article-level units as the default **rule node**
- preserve child nodes for clause, point, and paragraph-level fallback
- fall back to clause or paragraph chunks only when the article is too dense or the query targets a specific sub-rule
- use sentence-level splitting only as a last resort
- keep parent-child provenance: `law -> article -> clause -> point -> source span`
- store normalized citation keys such as `(law_id, aid)` for evaluation

**Real Reasoning Activation Plan**

Goal: turn the current baseline into a real LegalGraphRAG system that produces meaningful `prediction`, `justification`, `law_evidence`, and `case_evidence` for every case.

Minimum requirements:

1. Open-weight SLM API backend

   Use one active local or self-hosted SLM under 10B parameters per run. The pipeline should call the model through an HTTP API instead of embedding model inference directly into retrieval/orchestration code.

   Good starting choices:

   ```text
   luanngo/Qwen3-4B-VietNamese-Legal-Chat
   Qwen2.5-7B-Instruct
   Qwen3-4B-Instruct
   Mistral-7B-Instruct
   Gemma-class <10B instruct model
   ```

   Supported API styles:

   ```text
   Ollama native /api/chat
   OpenAI-compatible /v1/chat/completions from vLLM, LM Studio, llama.cpp server, or similar self-hosted runtime
   ```

   Do not use proprietary hosted model APIs for competition inference.

2. Environment configuration

   Create a local `.env` from `env.example` and set:

   ```text
   ALQAC_LLM_PROVIDER=ollama
   ALQAC_LLM_BASE_URL=http://localhost:11434
   ALQAC_LLM_MODEL_NAME=qwen2.5:7b-instruct
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<local password>
   NEO4J_DATABASE=neo4j
   ALQAC_TEAM_TOKEN=<official team token>
   ALQAC_API_URL=https://alqac-api.ngrok.pro
   FAILURE_BACKOFF_SECONDS=2.0
   ```

3. Legal graph build

   Build the graph from the legal corpus and persist it to Neo4j with three layers:

   - Ontology layer: legal concepts, dispute types, remedies, party roles, liability elements, relation aliases
   - Fact layer: case facts, claims, defenses, requested remedies, accepted/rejected amounts, factual patterns
   - Rule layer: laws, articles, clauses, points, legal rules, exceptions, procedural rules, fee rules

   Each node should keep provenance back to the original source text. Local JSONL graph files may be written as debug/cache artifacts, but Neo4j is the primary runtime graph store.

   Minimum node schema:

   ```json
   {
     "node_id": "rule:91/2015/QH13:584",
     "layer": "rule",
     "node_type": "article|clause|point|concept|fact_pattern|issue",
     "law_id": "91/2015/QH13",
     "aid": 584,
     "text": "source text or normalized label",
     "source_chunk_id": "chunk_...",
     "parent_id": "rule:91/2015/QH13",
     "aliases": ["bồi thường thiệt hại ngoài hợp đồng"]
   }
   ```

   Neo4j labels:

   ```text
   :Law
   :Article
   :Clause
   :Point
   :Concept
   :Issue
   :FactPattern
   ```

   Minimum edge schema:

   ```json
   {
     "src": "ontology:liability:animal_damage",
     "dst": "rule:91/2015/QH13:603",
     "edge_type": "governed_by|cites|contains|related_to|exception_to|analogous_to",
     "weight": 1.0,
     "evidence": "matched legal marker, citation, or extracted relation"
   }
   ```

   Neo4j relationship types:

   ```text
   CONTAINS
   CITES
   GOVERNED_BY
   RELATED_TO
   EXCEPTION_TO
   ANALOGOUS_TO
   MATCHES_FACT_PATTERN
   ```

   Required Neo4j constraints and indexes:

   ```cypher
   CREATE CONSTRAINT law_id_unique IF NOT EXISTS
   FOR (n:Law) REQUIRE n.law_id IS UNIQUE;

   CREATE CONSTRAINT node_id_unique IF NOT EXISTS
   FOR (n:LegalNode) REQUIRE n.node_id IS UNIQUE;

   CREATE INDEX article_lookup IF NOT EXISTS
   FOR (n:Article) ON (n.law_id, n.aid);

   CREATE INDEX concept_alias IF NOT EXISTS
   FOR (n:Concept) ON (n.normalized_alias);
   ```

4. Agent pipeline

   Use the LegalGraphRAG flow:

   - Researcher: retrieve candidate evidence chains across graph layers
   - Auditor: verify evidence against the source corpus
   - Adjudicator: produce the final label using only verified evidence

5. Output contract

   `experiments/run_v0_baseline.json` must contain one object per case:

   ```json
   {
     "case_id": "case_4101",
     "prediction": "A_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN|B_WIN",
     "confidence": 0.0,
     "justification": "Concise evidence-grounded reason.",
     "law_evidence": [{"law_id": "91/2015/QH13", "aid": 584}],
     "case_evidence": ["case_4101_chunk_..."],
     "graph_path": ["ontology:...", "fact:...", "rule:..."],
     "api_calls": 1
   }
   ```

6. Pass/fail checks

   A run is considered valid only if:

   - `--dry-run` is not used
   - retrieved evidence includes graph-grounded paths
   - Auditor removes unsupported evidence
   - every case has exactly one valid label
   - invalid JSON is handled by retry or fallback
   - no proprietary model/API is used in inference

**Implementation Gaps**

These are the immediate gaps between the current baseline and a real LegalGraphRAG submission candidate:

| Gap | Current state | Required next step |
|---|---|---|
| Graph structure | flat chunks and lexical/dense retrieval | build ontology/fact/rule graph with provenance |
| Evidence verification | heuristic citation usefulness filter | add Auditor-style source verification |
| Reasoning | direct prompt-based classification | add Researcher -> Auditor -> Adjudicator flow |
| Retrieval quality | hybrid retrieval over chunks | graph-aware retrieval plus neighbor expansion |
| Output traceability | chunk ids only | include graph paths and source-backed evidence chains |
| JSON robustness | parser fallback exists | log invalid JSON rate and retry if needed |

**LegalGraphRAG Retrieval Design**

Use the current hybrid index as a **seed generator**. LegalGraphRAG retrieval should not stop at top-k similarity results.

Target retrieval stages:

```text
Question
  ->
Query Analyzer
  - normalize Vietnamese legal terms and aliases
  - extract issue, party roles, claim type, remedy, dates, cited statutes
  - decide layer weights: ontology / fact / rule
  ->
Seed Retrieval
  - BM25 over article/clause text
  - dense retrieval over article/clause text
  - exact citation lookup for "Điều X", statute names, and law_id aliases
  - optional case-evidence retrieval from official case API
  ->
Seed-to-Graph Mapping
  - map chunk_id to rule node in Neo4j
  - map extracted issue/remedy to ontology nodes in Neo4j
  - map case facts to fact-pattern nodes in Neo4j
  ->
Graph Expansion
  - run Cypher traversal over parent/child: law -> article -> clause -> point
  - expand citations: article cites article or related law
  - expand ontology: issue -> liability elements -> matching rules
  - expand fact pattern: factual pattern -> prior outcome -> supporting rules
  - limit expansion depth to 1 for fast mode, 2 for deep mode
  ->
Evidence Chain Builder
  - create chains such as issue -> concept -> rule article -> supporting clause
  - attach source chunks and `(law_id, aid)` to every rule node
  - remove chains with no source-backed law evidence
  ->
Score Fusion
  - combine BM25, dense, exact-citation, graph distance, edge weight, and Auditor score
  - prefer direct rule matches over broad procedural/fee-only matches
  - downweight deprecated or unrelated procedural evidence
  ->
Rerank and Compress
  - rerank chains, not individual chunks
  - deduplicate by `(law_id, aid)` and keep the best source span
  - output top evidence chains for Auditor
```

Recommended scoring:

```text
final_score =
  0.25 * bm25_norm
  + 0.25 * dense_norm
  + 0.20 * graph_score
  + 0.15 * exact_citation_score
  + 0.10 * legal_issue_match
  + 0.05 * freshness_or_non_deprecated_score
```

Graph score:

```text
graph_score = edge_weight * (1 / (1 + graph_distance))
```

Neo4j traversal query pattern:

```cypher
MATCH (seed:LegalNode {node_id: $seed_node_id})
MATCH path = (seed)-[r:CONTAINS|CITES|GOVERNED_BY|RELATED_TO|EXCEPTION_TO|ANALOGOUS_TO*1..$depth]-(candidate:LegalNode)
WHERE candidate.layer IN $target_layers
RETURN path, candidate
LIMIT $limit
```

Evidence chain contract:

```json
{
  "chain_id": "chain_case_4101_001",
  "graph_path": [
    "ontology:issue:tort_damage",
    "ontology:liability:animal_damage",
    "rule:91/2015/QH13:603"
  ],
  "source_chunks": ["chunk_..."],
  "law_evidence": [{"law_id": "91/2015/QH13", "aid": 603}],
  "score": 0.82,
  "why": "animal-caused damage issue maps directly to Civil Code article 603"
}
```

Retrieval acceptance criteria:

- every returned chain must contain at least one source-backed law node
- every `law_evidence` item must map to an active chunk in `data/chunks.jsonl`
- graph expansion must never invent law articles missing from the corpus
- every graph path must be reproducible from Neo4j by `node_id`
- procedural and court-fee rules should not dominate unless the query is procedural or fee-related
- top-k should include both substantive rules and necessary procedural rules when both are relevant
- evaluation should use `data/retrieval_gold.json` and report hit@k, recall@k, MRR, and evidence-chain precision

**Work Split**

**1. Khoa - Lead, Orchestration, DevOps**

Main responsibility: make the whole system runnable, measurable, and submission-ready.

Deliverables:

- repo structure
- Neo4j service/config and graph build pipeline
- unified config system
- SLM API inference pipeline
- submission CLI with rate limiting
- experiment logging
- leaderboard tracker

Priority tasks:

1. Create pipeline modules around LegalGraphRAG:

```text
graph_construct/
retrieval/
reasoning/
verification/
evaluation/
orchestration/
experiments/
```

2. Build the end-to-end flow:

```text
load question -> hybrid seed retrieve -> Neo4j graph retrieve -> researcher -> auditor -> adjudicator via SLM API -> verify -> output
```

3. Build submission guardrails:

- max 3 submissions/day
- queue API calls at 5-second intervals
- save every submitted version with config hash

**2. Hung - Graph and Retrieval Engineer**

Main responsibility: build the hierarchical legal graph and make retrieval evidence-aware.

Deliverables:

- corpus cleaner
- node/edge schema
- Neo4j graph importer
- multi-granular chunker
- Cypher graph traversal retriever
- hybrid BM25 + dense retrieval
- reranker
- retrieval evaluation report

Recommended retrieval pipeline:

```text
Question
  ->
legal query normalization
  ->
issue / claim / remedy / citation extraction
  ->
graph layer routing
  ->
exact citation lookup
  ->
BM25 top 80 + dense top 80 over article/clause nodes
  ->
seed-to-Neo4j node mapping
  ->
Cypher graph expansion from seed nodes
  ->
evidence chain construction
  ->
chain score fusion
  ->
rerank top 20 chains
  ->
Auditor-ready evidence chains top 8
```

Start with:

```text
bm25_weight = 0.25
dense_weight = 0.25
graph_weight = 0.20
exact_citation_weight = 0.15
legal_issue_weight = 0.10
freshness_weight = 0.05
article-level chunks as default
clause/point fallback when article is too dense or query targets sub-rules
overlap = 80-120 tokens
seed_top_k = 80 per retriever
graph_expansion_depth = 1 fast mode, 2 deep mode
top_k after chain rerank = 8
```

Important: retrieval evaluation should measure whether the evidence chain is actually useful, not just semantically similar.

**3. Thinh - Reasoning and Verification Engineer**

Main responsibility: turn verified evidence into the right 4-label answer.

Deliverables:

- label definition document
- Researcher prompt
- Auditor prompt
- Adjudicator prompt
- SLM API request/response schema
- verifier prompt
- prompt A/B variants
- error taxonomy

Immediate priority:

Analyze 20-50 public samples and define the 4 labels exactly. This is the highest-risk unknown.

Prompt structure:

```text
Input:
- Question
- Retrieved graph paths
- Verified legal passages
- Candidate label definitions

Tasks:
1. Identify the relevant legal issue.
2. Keep only evidence verified by the Auditor.
3. Apply the law to the facts.
4. Choose exactly one label.
5. Return JSON.
```

Output format:

```json
{
  "label": "LABEL_X",
  "confidence": 0.0,
  "evidence_ids": ["doc_1", "doc_3"],
  "justification": "Concise statute-grounded reason."
}
```

Use two modes:

- Fast mode: direct classification when graph evidence is strong
- Deep mode: multi-step legal analysis when evidence conflicts or confidence is low

Do not rely on exposed verbose chain-of-thought in final output. Keep internal reasoning structured, but output concise evidence-grounded justification.

**4. Tuan Anh - QA, Evaluation, Leaderboard**

Main responsibility: know whether changes actually improve the system.

Deliverables:

- local evaluator
- baseline report
- confusion matrix
- per-label metrics
- retrieval diagnostics
- A/B testing dashboard or CSV
- leaderboard submission log

Evaluation layers:

```text
1. Retrieval metrics
   - recall@5
   - MRR
   - evidence-chain usefulness

2. Classification metrics
   - accuracy
   - macro F1
   - per-label precision/recall
   - confusion matrix

3. System metrics
   - latency
   - failure rate
   - invalid JSON rate
   - low-confidence rate

4. Leaderboard metrics
   - submission version
   - public score
   - delta vs previous
```

A/B testing rule:

Only change one major variable per run:

- prompt only
- graph layer routing only
- retriever only
- verifier only
- model only

Otherwise leaderboard feedback becomes hard to interpret.

**Implementation Timeline**

**Phase 0: Day 1 - Label and Data Recon**

Critical tasks:

- download public test
- identify exact 4 labels
- inspect 20-50 samples manually
- define output schema
- build first dummy baseline

Owner split:

- Khoa: repo + config + pipeline skeleton
- Hung: corpus inspection + graph schema draft
- Thinh: label analysis + prompt v0
- Tuan Anh: evaluation harness + baseline metrics

**Phase 1: Days 2-4 - Baseline LegalGraphRAG**

Goal: working end-to-end system.

Build:

- Neo4j graph construction and import
- graph-aware retriever with Cypher traversal
- simple hybrid fusion
- one open-weight SLM exposed through HTTP API
- one prompt
- JSON output parser
- local evaluator

First submission only after local sanity checks.

**Phase 2: Days 5-8 - Graph Quality Push**

Goal: improve evidence quality.

Add:

- auditor verification
- graph neighbor expansion
- Cypher query templates by legal issue type
- legal document routing
- citation usefulness filter
- statute/date normalization
- deprecated-law filtering if corpus supports it

This phase should get most effort. Bad retrieval will cap reasoning accuracy.

**Phase 3: Days 9-12 - Reasoning and Verification**

Goal: reduce label confusion.

Add:

- fast/deep reasoning modes
- verifier pass
- fallback for low-confidence cases
- few-shot examples selected from public samples
- prompt variants by legal domain or question type

**Phase 4: Days 13+ - Submission Optimization**

Goal: disciplined leaderboard gains.

Daily loop:

```text
Morning:
- run local full evaluation
- inspect top failure modes

Afternoon:
- make one targeted change
- run A/B comparison

Evening:
- use at most one leaderboard submission if local result improves
```

Keep 1-2 submissions reserved near the deadline.

**Model Strategy**

Start with one reliable `<10B` SLM exposed through an API endpoint and avoid model-hopping early. A single run should use one active SLM backend; the candidate list below is for A/B experiments, not ensemble inference.

Recommended practical path:

```text
Primary LLM candidates under 10B:
- luanngo/Qwen3-4B-VietNamese-Legal-Chat
- Qwen2.5-7B-Instruct
- Qwen3-4B-Instruct
- Mistral-7B-Instruct
- Gemma-2-9B-It / Gemma-class <10B instruct model

Model roles:
- Vietnamese legal specialist: luanngo/Qwen3-4B-VietNamese-Legal-Chat
- general reasoning fallback: Qwen2.5-7B-Instruct
- compact fast path: Qwen3-4B-Instruct or Qwen3-4B-VietNamese-Legal-Chat
- heavier open-weight fallback: Mistral-7B-Instruct or Gemma-2-9B-It

Embedding: Vietnamese/multilingual embedding model
Reranker: multilingual cross-encoder if latency allows
```

SLM API contract:

```text
Pipeline sends:
- case_id
- case_query
- verified evidence chains
- compact law_evidence and case_evidence
- strict output schema

SLM returns JSON only:
- prediction
- confidence
- evidence_ids
- justification
```

Use the LegalGraphRAG idea carefully: the graph and verification layers should carry much of the burden. Do not spend the first week on heavy continual pretraining unless the base pipeline is already stable.

**Risks**

Highest risks:

1. Misunderstanding the 4 labels
   Mitigation: Thinh analyzes public samples immediately.

2. Graph retrieval returns related but non-answering evidence
   Mitigation: Hung builds graph routing plus Auditor verification.

3. Overfitting public test
   Mitigation: Tuan Anh tracks per-label failures and avoids tuning only to visible examples.

4. Too many uncontrolled experiments
   Mitigation: Khoa enforces config/version logging.

5. Using disallowed proprietary tools
   Mitigation: competition system must use only open-weight models.

**Immediate Next Actions**

1. Khoa: create repo structure, Neo4j service/config, graph pipeline skeleton, submission tracker.
2. Hung: build graph schema, Neo4j importer, and graph-aware retrieval baseline.
3. Thinh: define exact 4 labels from public samples.
4. Tuan Anh: implement evaluator, retrieval_gold benchmark, and baseline report.
5. Team: run first end-to-end LegalGraphRAG baseline before optimizing anything.
