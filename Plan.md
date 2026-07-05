**Core Strategy**

Build an **open-weight Agentic RAG legal QA system** under the ALQAC constraints:

- Model: open-weight, `<10B`
- No ChatGPT / Claude / Gemini in the competition system
- Focus: retrieval quality first, reasoning second
- Optimize through public test analysis, A/B runs, and careful leaderboard submissions
- Use the VLSP 2025 methodology ideas as system components, not as a direct copy of their training pipeline

Key transfer from VLSP Section 3:

- Continual legal-domain adaptation → use legal corpus cleaning, statute filtering, and embedding/domain adaptation where allowed
- Multi-task reasoning → split the system into citation usefulness, document routing, MCQ/label classification, and final answer reasoning
- Reasoning vs non-reasoning modes → use deeper reasoning only when needed; use fast direct classification for easy cases

**Recommended Architecture**

```text
Question
  ↓
Query Analyzer
  - detect legal domain
  - extract entities, dates, statutes, issue type
  - decide retrieval depth
  ↓
Retrieval Agent
  - BM25 lexical search
  - dense vector search
  - hybrid fusion
  - reranking
  ↓
Citation Usefulness Agent
  - decide whether retrieved law actually answers the question
  - filter weak or irrelevant evidence
  ↓
Reasoning Agent
  - apply facts to statutes
  - classify into 4 labels
  - produce concise justification
  ↓
Verifier
  - check label/evidence consistency
  - fallback if confidence is low
  ↓
Submission Output
```

**Work Split**

**1. Khoa — Lead, Orchestration, DevOps**

Main responsibility: make the whole system runnable, measurable, and submission-ready.

Deliverables:

- GitHub repo structure
- Supabase / pgvector or local FAISS setup
- unified config system
- inference pipeline
- submission CLI with 1 request / 5 seconds rate limiting
- experiment logging
- leaderboard tracker

Priority tasks:

1. Create repo and folders:

```text
data/
retrieval/
reasoning/
evaluation/
orchestration/
experiments/
leaderboard/
```

2. Build `run_pipeline.py` that connects:

```text
load question → retrieve → rerank → reason → verify → output
```

3. Build submission guardrails:

- max 3 submissions/day
- queue API calls at 5-second intervals
- save every submitted version with config hash

4. Maintain experiment matrix:

```text
model
embedding
chunk size
top_k
reranker
prompt version
score
latency
notes
```

**2. Hưng — Retrieval Engineer**

Main responsibility: make the system find the right law.

Use ideas from VLSP:

- clean statutory corpus
- remove outdated/repealed legal documents where possible
- support document routing
- evaluate citation usefulness, not just similarity

Deliverables:

- corpus cleaner
- chunker
- BM25 index
- dense vector index
- hybrid retriever
- reranker
- retrieval evaluation report

Recommended retrieval pipeline:

```text
Question
  ↓
legal query normalization
  ↓
BM25 top 50
  +
dense top 50
  ↓
score fusion
  ↓
reranker top 10
  ↓
citation usefulness filter top 3-5
```

Hybrid score:

```text
score = alpha * BM25 + (1 - alpha) * dense_similarity
```

Start with:

```text
alpha = 0.45
chunk size = 350-600 tokens
overlap = 80-120 tokens
top_k before rerank = 50
top_k after rerank = 5
```

Important: retrieval evaluation should not only check “similar-looking” documents. It should check whether the retrieved citation is actually useful for answering the question, following the VLSP citation usefulness task.

**3. Thịnh — Reasoning Engineer**

Main responsibility: turn retrieved evidence into the right 4-label answer.

Use ideas from VLSP:

- separate reasoning and non-reasoning modes
- build instruction-style prompts
- use citation usefulness before final reasoning
- support document detection / legal domain routing

Deliverables:

- prompt templates
- label definition document
- reasoning agent
- verifier prompt
- prompt A/B variants
- error taxonomy

Immediate priority:

Analyze 20-50 public samples and define the 4 labels exactly. This is the highest-risk unknown.

Prompt structure:

```text
Input:
- Question
- Retrieved legal passages
- Candidate label definitions

Tasks:
1. Identify the relevant legal issue.
2. Select only useful legal evidence.
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

- Fast mode: direct classification when retrieval confidence is high
- Deep mode: multi-step legal analysis when evidence conflicts or confidence is low

Do not rely on exposed verbose chain-of-thought in final output. Keep internal reasoning structured, but output concise evidence-grounded justification.

**4. Tuấn Anh — QA, Evaluation, Leaderboard**

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
   - citation usefulness precision

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
- retriever only
- reranker only
- model only
- chunking only

Otherwise leaderboard feedback becomes hard to interpret.

**Implementation Timeline**

**Phase 0: Day 1 — Label and Data Recon**

Critical tasks:

- Download public test
- Identify exact 4 labels
- Inspect 20-50 samples manually
- Define output schema
- Build first dummy baseline

Owner split:

- Khoa: repo + config + pipeline skeleton
- Hưng: corpus inspection + chunking draft
- Thịnh: label analysis + prompt v0
- Tuấn Anh: evaluation harness + baseline metrics

**Phase 1: Days 2-4 — Baseline Agentic RAG**

Goal: working end-to-end system.

Build:

- BM25 retriever
- dense retriever
- simple hybrid fusion
- one open-weight model
- one prompt
- JSON output parser
- local evaluator

First submission only after local sanity checks.

**Phase 2: Days 5-8 — Retrieval Quality Push**

Goal: improve evidence quality.

Add:

- reranker
- query decomposition
- legal document routing
- citation usefulness filter
- statute/date normalization
- deprecated-law filtering if corpus supports it

This phase should get most effort. Bad retrieval will cap reasoning accuracy.

**Phase 3: Days 9-12 — Reasoning and Verification**

Goal: reduce label confusion.

Add:

- fast/deep reasoning modes
- verifier pass
- fallback for low-confidence cases
- few-shot examples selected from public samples
- prompt variants by legal domain or question type

**Phase 4: Days 13+ — Submission Optimization**

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

Start with one reliable `<10B` model and avoid model-hopping early.

Recommended practical path:

```text
Primary LLM: Qwen / Mistral-class 7B or smaller open-weight instruct model
Embedding: Vietnamese/multilingual legal-capable embedding model
Reranker: multilingual cross-encoder if latency allows
```

Use VLSP’s Qwen3-style idea carefully: small models can work if retrieval and task decomposition are strong. Do not spend the first week on heavy continual pretraining unless the base pipeline is already stable.

**Risks**

Highest risks:

1. Misunderstanding the 4 labels
   Mitigation: Thịnh analyzes public samples immediately.

2. Retrieval returns legally related but non-answering statutes
   Mitigation: Hưng builds citation usefulness filtering.

3. Overfitting public test
   Mitigation: Tuấn Anh tracks per-label failures and avoids tuning only to visible examples.

4. Too many uncontrolled experiments
   Mitigation: Khoa enforces config/version logging.

5. Using disallowed proprietary tools
   Mitigation: competition system must use only open-weight models; avoid Gemini-style synthetic generation from the VLSP paper unless rules explicitly allow external generation.

**Immediate Next Actions**

1. Khoa: create repo structure, pipeline skeleton, submission tracker.
2. Hưng: build BM25 baseline and inspect corpus/chunk strategy.
3. Thịnh: define exact 4 labels from public samples.
4. Tuấn Anh: implement evaluator and baseline report.
5. Team: run first end-to-end baseline before optimizing anything.
