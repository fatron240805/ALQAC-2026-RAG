# Retrieval Guide

Tài liệu mô tả cách cài đặt, cấu hình và sử dụng module retrieval của project
ALQAC 2026 RAG theo code hiện tại.

## 1. Kiến trúc

~~~text
data/chunks.jsonl
  -> data_adapters
  -> HybridIndexer (BGE-M3 sparse + dense)
  -> DocumentRouter
  -> Reranker
  -> Citation filter
  -> RetrievalService
  -> EvidenceChain cho reasoning
~~~

Backend runtime:

- Có graph retriever và Neo4j hoạt động: neo4j_graph.
- Neo4j không sẵn sàng và không truyền require-graph: fallback flat_hybrid.
- use_bge_m3=True ưu tiên BAAI/bge-m3.
- BGE-M3 lỗi thì thử SentenceTransformer nếu được bật, sau đó dùng hash vector.

## 2. Yêu cầu môi trường

- Windows PowerShell hoặc shell tương đương.
- Python 3.12.x và Git.
- data/chunks.jsonl.
- data/ALQAC2026_public_test.json khi chạy pipeline/benchmark.
- Internet lần đầu để tải BGE-M3 và BGE reranker.
- NVIDIA GPU không bắt buộc nhưng được khuyến nghị.

Với device=auto, code chọn cuda:0 nếu torch.cuda.is_available() là True,
nếu không dùng CPU:

~~~powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
~~~

## 3. Cài virtual environment và thư viện

~~~powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Dependencies trong requirements.txt:

| Thư viện | Vai trò |
|---|---|
| numpy | vector và score fusion |
| torch | CPU/GPU runtime |
| FlagEmbedding | BGEM3FlagModel và FlagReranker |
| sentence-transformers | fallback dense embedding |
| rank-bm25 | fallback sparse BM25 |
| neo4j | driver Neo4j/AuraDB |
| requests | HTTP clients |
| python-dotenv | tiện ích cho script tự load .env; pipeline chưa tự gọi load_dotenv |

Import model:

~~~python
from FlagEmbedding import BGEM3FlagModel, FlagReranker
~~~

## 4. Tài khoản và dịch vụ

Flat retrieval local không cần account bên ngoài sau khi model đã tải.

| Thành phần | Cần gì | Biến/cấu hình |
|---|---|---|
| BGE-M3/BGE reranker | Model public, Internet lần đầu | embedding_model, reranker_model_name |
| Neo4j local | Service đang chạy | NEO4J_URI=bolt://localhost:7687 |
| Neo4j AuraDB | Instance và credentials Aura | NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD |
| Ollama | Server và model local | ALQAC_LLM_* |
| vLLM/LM Studio/llama.cpp | Endpoint OpenAI-compatible | ALQAC_LLM_PROVIDER |
| ALQAC case API | Team token do ban tổ chức cấp | ALQAC_TEAM_TOKEN, ALQAC_API_URL |

Không cần OpenAI API key nếu dùng endpoint local. Không cần Neo4j cho flat
retrieval. Không cần ALQAC_TEAM_TOKEN cho RetrievalService.retrieve(); token chỉ
được CaseRetrievalClient dùng ở endpoint /retrieve.

CaseRetrievalClient.from_env() ưu tiên ALQAC_RETRIEVAL_API_BASE_URL, sau đó dùng
ALQAC_API_URL. Mỗi request /retrieve là một call có rate limit; pipeline dùng
RateLimiter theo SECONDS_BETWEEN_API_CALLS và không nên gọi song song cùng team
token.

## 5. Biến môi trường

Tạo secret file local và không commit file này:

~~~powershell
Copy-Item env.example .env
~~~

Mẫu LLM và API:

~~~text
ALQAC_LLM_PROVIDER=ollama
ALQAC_LLM_BASE_URL=http://localhost:11434
ALQAC_LLM_MODEL_NAME=qwen2.5:7b-instruct
ALQAC_LLM_TIMEOUT_SECONDS=300
ALQAC_LLM_MAX_RETRIES=2
# Chỉ dùng nếu endpoint OpenAI-compatible có authentication.
ALQAC_LLM_API_KEY=replace_with_llm_endpoint_key

ALQAC_TEAM_TOKEN=replace_with_team_token
ALQAC_API_URL=https://alqac-api.ngrok.pro
SECONDS_BETWEEN_API_CALLS=5.0
FAILURE_BACKOFF_SECONDS=2.0
~~~

Neo4j local:

~~~text
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace_with_local_password
NEO4J_DATABASE=neo4j
NEO4J_MAX_CONNECTION_POOL_SIZE=8
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=30
NEO4J_IMPORT_BATCH_SIZE=200
NEO4J_ERROR_LOG_PATH=logs/neo4j_errors.log
~~~

Neo4j Aura:

~~~text
NEO4J_URI=neo4j+s://<aura-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace_with_aura_password
NEO4J_DATABASE=
~~~

Các scheme hỗ trợ: neo4j, neo4j+s, neo4j+ssc, bolt, bolt+s và bolt+ssc.
Error log mặc định ở logs/neo4j_errors.log và URI/password được redact.

Code hiện đọc os.environ, chưa tự gọi load_dotenv(). Cài python-dotenv không
tự làm biến .env xuất hiện trong process. Export thủ công trước khi chạy:

~~~powershell
$env:ALQAC_LLM_PROVIDER = "ollama"
$env:ALQAC_LLM_BASE_URL = "http://localhost:11434"
$env:ALQAC_LLM_MODEL_NAME = "qwen2.5:7b-instruct"
$env:NEO4J_URI = "neo4j+s://<instance>.databases.neo4j.io"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "<password>"
$env:NEO4J_DATABASE = ""
~~~

## 6. Dữ liệu và mapping

data/chunks.jsonl có một JSON object mỗi dòng. Field tối thiểu:

~~~json
{
  "chunk_id": "chunk_...",
  "doc_id": "doc_...",
  "text": "Nội dung điều/khoản/điểm...",
  "law_id": "47/2010/QH12",
  "aid": 270,
  "article_number": "270",
  "article_index": 1,
  "unit_path": "47/2010/QH12 Điều 270",
  "deprecated": false
}
~~~

chunk_to_indexer_doc() đổi thành doc_id=chunk_id, content=text và metadata chứa
law_id, aid, article_number, article_index, unit_path và các field legal khác.
source_chunk_id được bổ sung/đọc ở graph candidate khi graph row map về chunk.
Do đó
HybridIndexer.doc_ids là chunk IDs, không phải document-level doc IDs. Graph
seed phải tìm node có source_chunk_id hoặc chunk_id trùng ID này.

EvidenceChain tạo law_evidence khi có law_id và aid; nếu aid trống, code fallback
sang article_number hoặc article_index. Mặc định
RetrievalServiceConfig.require_law_evidence=True; candidate thiếu mapping bị
loại và tính vào trace.stats.dropped_unmapped_candidates. Debug có thể dùng
require_law_evidence=False.


## 7. Bật graph-aware retrieval

### 10.1. Sinh graph artifact

~~~powershell
python -m orchestration.run_pipeline build-graph
~~~

Artifact ở data/graph:

- nodes.jsonl.
- edges.jsonl.
- graph_manifest.json.

build_graph_records() bỏ chunk deprecated, tạo LegalNode theo layer law, rule,
ontology, fact và các cạnh CONTAINS, CITES, GOVERNED_BY, RELATED_TO,
EXCEPTION_TO, ANALOGOUS_TO, MATCHES_FACT_PATTERN khi có.

### 10.2. Import vào Neo4j

~~~powershell
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
python -m orchestration.run_pipeline build-graph --import-neo4j --batch-size 200
~~~

Chỉ dùng clear-graph khi muốn xóa toàn bộ LegalNode:

~~~powershell
python -m orchestration.run_pipeline build-graph --import-neo4j --clear-graph --batch-size 200
~~~

Import dùng MERGE theo node_id, chia batch bounded và ghi Neo4j error log. Aura
Free nên bắt đầu với batch 200 hoặc nhỏ hơn.

### 10.3. GraphRetriever thực hiện gì?

LegalGraphRetriever.retrieve(query):

1. Lấy seed_top_k candidate từ HybridIndexer.
2. Phân tích query và document routing.
3. Rerank seed candidate.
4. Dùng seed chunk_id tìm node qua source_chunk_id/chunk_id.
5. Traverse depth 1 hoặc 2; query có citation hoặc dài sẽ dùng deep.
6. Giữ layer rule, ontology, fact.
7. Tạo fused graph score từ seed relevance, graph distance, citation, issue
   match và freshness.
8. Chọn ontology community rồi chạy community-cluster rerank.
9. Merge seed và graph candidate, giữ graph quota theo graph_candidate_ratio.
10. Chạy citation filter và bảo toàn graph path hợp lệ.

Nếu Neo4j có node nhưng không có source_chunk_id/chunk_id trùng index, bước 4
trả rỗng và graph traversal không tăng recall.

### 10.4. Dùng graph retriever trong Python

~~~python
from orchestration.config import PipelineConfig
from orchestration.run_pipeline import build_graph_retriever, load_or_build_index
from retrieval import RetrievalService, RetrievalServiceConfig

config = PipelineConfig()
indexer = load_or_build_index(config)
graph_retriever = build_graph_retriever(config, indexer, require_graph=True)

service = RetrievalService(
    indexer,
    graph_retriever=graph_retriever,
    config=RetrievalServiceConfig(
        seed_top_k=config.top_k_before_rerank,
        final_top_k=config.top_k_after_rerank,
        hybrid_alpha=config.hybrid_alpha,
    ),
)
result = service.retrieve(query, top_k=8)
print(result.trace.as_dict())
~~~

Statistic cần theo dõi:

~~~text
seed_count
graph_expansion_used
graph_candidates_before_rerank
merged_candidate_count
final_graph_candidate_count
dropped_unmapped_candidates
~~~

## 8. PipelineConfig và JSON config

PipelineConfig() có default. Có thể tạo configs/retrieval_local.json:

~~~json
{
  "embedding_model": "BAAI/bge-m3",
  "use_bge_m3": true,
  "use_sentence_transformer": false,
  "retrieval_device": "auto",
  "retrieval_batch_size": 12,
  "retrieval_max_length": 1024,
  "top_k_before_rerank": 80,
  "top_k_after_rerank": 8,
  "hybrid_alpha": 0.50,
  "use_graph_retrieval": true,
  "graph_expansion_depth_fast": 1,
  "graph_expansion_depth_deep": 2,
  "graph_bm25_weight": 0.25,
  "graph_dense_weight": 0.25,
  "graph_weight": 0.20,
  "graph_exact_citation_weight": 0.15,
  "graph_legal_issue_weight": 0.10,
  "graph_freshness_weight": 0.05,
  "graph_community_top_k": 1,
  "graph_community_member_top_k": 20,
  "graph_candidate_ratio": 0.25,
  "graph_reranker_method": "community_cluster",
  "graph_backend": "neo4j",
  "reranker_model_name": "BAAI/bge-reranker-v2-m3",
  "use_gpu_reranker": true,
  "reranker_device": "auto",
  "reranker_batch_size": 16,
  "chunks_path": "data/chunks.jsonl",
  "index_path": "data/index",
  "graph_path": "data/graph",
  "public_test_path": "data/ALQAC2026_public_test.json",
  "prompt_path": "reasoning/prompts/prompt_v0.md",
  "gold_path": "data/local_validation_gold.json"
}
~~~

Chạy config:

~~~powershell
python -m orchestration.run_pipeline --config configs/retrieval_local.json build-index
python -m orchestration.run_pipeline --config configs/retrieval_local.json build-graph
~~~

PipelineConfig.from_file() quy đổi path tương đối theo project root. Key lạ
được giữ trong config.extra nhưng không tự tác động đến retrieval.

### 11.1. Các knob thường điều chỉnh

| Field | Tác động |
|---|---|
| top_k_before_rerank | Recall seed; tăng RAM/thời gian rerank và graph |
| top_k_after_rerank | Số evidence sang reasoning |
| hybrid_alpha | 0 thiên dense, 1 thiên sparse |
| retrieval_batch_size | Tốc độ/VRAM khi build BGE |
| retrieval_max_length | Context model và VRAM |
| use_bge_m3 | Bật/tắt BGE-M3 |
| use_sentence_transformer | Fallback sau khi BGE không load |
| retrieval_device | auto, cpu, cuda, cuda:0 |
| reranker_device | device BGE reranker |
| use_gpu_reranker | Cho phép load FlagReranker |
| graph_expansion_depth_fast/deep | Độ sâu traversal |
| graph_candidate_ratio | Tỷ lệ slot cho graph candidate |
| graph_community_top_k | Số ontology community ưu tiên |
| graph_community_member_top_k | Số member tối đa mỗi community |
| graph_reranker_method | community_cluster để dùng cluster reranker |
| neo4j_import_batch_size | Batch import mặc định |

## 9. Full pipeline và runtime checks

### 12.1. Dry run offline

~~~powershell
python -m orchestration.run_pipeline run --dry-run --limit 1 --rebuild-index
python -m orchestration.run_pipeline validate-output
~~~

Dry run không gọi LLM thật, ALQAC API hoặc graph live.

### 12.2. Runtime check

~~~powershell
python -m orchestration.run_pipeline check-runtime
python -m orchestration.run_pipeline check-runtime --ping-llm
python -m orchestration.run_pipeline check-runtime --graph-only --ping-neo4j
~~~

Ping ALQAC case API là request có quota, chỉ chạy khi cần:

~~~powershell
python -m orchestration.run_pipeline check-runtime --ping-case-api
~~~

### 12.3. Ollama

~~~powershell
ollama pull qwen2.5:7b-instruct
ollama list
$env:ALQAC_LLM_PROVIDER = "ollama"
$env:ALQAC_LLM_BASE_URL = "http://localhost:11434"
$env:ALQAC_LLM_MODEL_NAME = "qwen2.5:7b-instruct"
python -m orchestration.run_pipeline check-runtime --ping-llm
~~~

### 12.4. Inference thật giới hạn

~~~powershell
python -m orchestration.run_pipeline run --limit 1 --rebuild-index --require-graph
python -m orchestration.run_pipeline validate-output --require-real
~~~

require-graph làm command fail nếu Neo4j không kết nối được hoặc graph rỗng.
Không truyền flag này thì pipeline có thể fallback flat.


## 10. Benchmark và chẩn đoán chất lượng

Graph benchmark mặc định dùng public test và gold
related_law_provisions. Report có cả end-to-end và graph traversal-only:

~~~powershell
python -m evaluation.retrieval_benchmark --retriever graph --top-k 1,3,5,8,30 --score-scope end_to_end --output experiments/retrieval_graph_benchmark_report.json --cases-output experiments/retrieval_graph_benchmark_cases.csv
~~~

Flat hybrid để đối chiếu:

~~~powershell
python -m evaluation.retrieval_benchmark --retriever hybrid --top-k 1,3,5,8,30
~~~

Diễn giải report:

- end_to_end: sau candidate selection, rerank và filter.
- graph_traversal_only: chỉ candidate từ graph traversal; metric thấp ở scope
  này chưa chắc toàn pipeline reasoning thấp.
- graph_expansion_used=False: thường do Neo4j lỗi, graph rỗng hoặc seed
  chunk_id không map source_chunk_id/chunk_id.
- dropped_unmapped_candidates cao: thiếu law_id hoặc aid trong metadata.

## 11. Troubleshooting

### 14.1. BGE-M3 không load

~~~powershell
python -c "from FlagEmbedding import BGEM3FlagModel; print('BGE-M3 import OK')"
python -c "import torch; print(torch.cuda.is_available())"
python -m orchestration.run_pipeline build-index
~~~

Kiểm tra data/index/index_meta.json. Giá trị mong đợi:

~~~json
{
  "model_name": "BAAI/bge-m3",
  "use_bge_m3": true,
  "retrieval_backend": "bge_m3",
  "device": "cuda:0"
}
~~~

### 14.2. Reranker GPU không chạy

Reranker không làm pipeline fail khi FlagReranker không load được; code fallback
deterministic. Để debug CPU:

~~~json
{
  "use_gpu_reranker": false,
  "reranker_device": "cpu"
}
~~~

### 14.3. Graph traversal không tăng

1. check-runtime --graph-only --ping-neo4j có legal_node_count > 0 không.
2. Graph có được build từ cùng data/chunks.jsonl với index không.
3. data/index/documents.jsonl có doc_id là chunk_id không.
4. Graph node có source_chunk_id hoặc chunk_id trùng index doc ID không.
5. Quan hệ có thuộc danh sách Neo4j store hỗ trợ không.
6. trace.stats.graph_candidates_before_rerank và
   trace.stats.final_graph_candidate_count có lớn hơn 0 không.

Kiểm tra mapping trực tiếp:

~~~python
from graph_construct.neo4j_store import Neo4jGraphStore

store = Neo4jGraphStore()
print(store.seed_nodes_for_chunk("chunk_id_can_kiem_tra"))
store.close()
~~~

### 14.4. Gold provision không map

Gold thường có law_id và aid, còn index có thể lưu field ở top-level hoặc dùng
article_number/article_index. Adapter cần giữ tối thiểu:

~~~text
metadata.law_id
metadata.aid
metadata.article_number
metadata.article_index
metadata.source_chunk_id
~~~

EvidenceChain lấy aid, fallback article_number hoặc article_index. Record thiếu
law_id hoặc số article hợp lệ bị xem là unmapped khi
require_law_evidence=True.

### 14.5. Neo4j lỗi

- Aura dùng neo4j+s://; local thường dùng bolt://.
- Không thêm quote vào giá trị .env.
- Đọc logs/neo4j_errors.log; URI/password đã được redact.
- Giảm NEO4J_IMPORT_BATCH_SIZE nếu Aura Free timeout.
- Không chạy đồng thời nhiều lệnh clear-graph.

### 14.6. ALQAC API lỗi

- 403: kiểm tra ALQAC_TEAM_TOKEN và header X-API-Key.
- 429: không chạy song song nhiều process cùng token; giữ ít nhất 5 giây/request.
- 503: database team tạm thời không sẵn sàng.
- Không lặp ping-case-api trong benchmark local.

## 12. Verification checklist

~~~powershell
python -m unittest discover -s tests -v
python -m py_compile retrieval\indexing.py retrieval\graph_retriever.py retrieval\reranker.py retrieval\service.py
python experiments\retrieval_smoke_test.py --flat --top-k 2 --content-chars 300
~~~

Smoke test tối thiểu cần có backend flat_hybrid hoặc neo4j_graph, returned_count
lớn hơn 0 và content của DOCUMENT 1. Graph mode nên có chain với graph_path
dạng node_a -> node_b. Nếu mode graph in flat_hybrid thì đó là fallback, chưa
phải graph retrieval thành công.

## 13. Checklist triển khai

- [ ] Python 3.12 trong .venv.
- [ ] pip install -r requirements.txt hoàn tất.
- [ ] torch.cuda.is_available() đúng mục tiêu CPU/GPU.
- [ ] chunks có chunk_id, text, law_id, aid.
- [ ] Index metadata đúng BGE-M3 nếu dùng BGE.
- [ ] graph artifact build từ cùng nguồn chunks với index.
- [ ] Neo4j có LegalNode và seed chunk IDs map được.
- [ ] Ping Neo4j thành công nếu dùng graph.
- [ ] LLM endpoint hoạt động nếu full inference.
- [ ] ALQAC_TEAM_TOKEN chỉ cấu hình khi cần case retrieval/submission.
- [ ] Đã chạy smoke test và benchmark trước public set.
- [ ] Không commit .env, password, API key hoặc token.



+
## 14. Quick start local flat retrieval

Build index:

~~~powershell
python -m orchestration.run_pipeline build-index
~~~

Index được lưu tại data/index:

- documents.jsonl: content và metadata chuẩn hóa.
- doc_ids.json: thứ tự IDs ứng với embedding matrix.
- dense_embeddings.npy: dense vectors.
- bm25_model.pkl: BGE lexical weights hoặc fallback BM25.
- index_meta.json: model, device, dimension và backend.

load_or_build_index() tự rebuild khi metadata không khớp use_bge_m3,
use_sentence_transformer hoặc embedding_model. Lệnh run có --rebuild-index để
ép rebuild.

Smoke test dùng query đầu tiên của public data:

~~~powershell
python experiments/retrieval_smoke_test.py --flat --top-k 5
~~~

Script in case ID, query, backend, reranker, setup_ms, retrieval_ms, số result,
score, legal evidence, source chunk, graph path, metadata và content.

Graph-aware mode, cho phép fallback:

~~~powershell
python experiments/retrieval_smoke_test.py --top-k 5
~~~

Bắt buộc graph và fail nếu Neo4j lỗi:

~~~powershell
python experiments/retrieval_smoke_test.py --require-graph --top-k 5
~~~

## 15. Public API RetrievalService

Đây là API nên dùng cho giai đoạn RAG kế tiếp:

~~~python
from orchestration.config import PipelineConfig
from orchestration.run_pipeline import load_or_build_index
from retrieval import RetrievalService, RetrievalServiceConfig
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.reranker import LexicalOverlapReranker

config = PipelineConfig()
indexer = load_or_build_index(config)

service = RetrievalService(
    indexer,
    reranker=LexicalOverlapReranker(
        use_gpu_reranker=config.use_gpu_reranker,
        reranker_model_name=config.reranker_model_name,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    ),
    citation_filter=HeuristicCitationUsefulnessFilter(
        max_results=config.top_k_after_rerank,
    ),
    config=RetrievalServiceConfig(
        seed_top_k=config.top_k_before_rerank,
        final_top_k=config.top_k_after_rerank,
        hybrid_alpha=config.hybrid_alpha,
    ),
)

result = service.retrieve(query, top_k=5, use_graph=False)
print(result.trace.as_dict())

for chain in result.chains:
    print(chain.rank, chain.doc_id, chain.law_evidence, chain.content)
~~~

### 8.1. RetrievalServiceConfig

~~~python
RetrievalServiceConfig(
    seed_top_k=80,             # candidate từ index trước rerank
    final_top_k=8,             # evidence tối đa trả về
    hybrid_alpha=0.50,         # sparse; dense = 1 - alpha
    require_law_evidence=True, # yêu cầu law_id + aid
)
~~~

### 8.2. RetrievalService.retrieve()

~~~python
result = service.retrieve(query, top_k=8, use_graph=None)
~~~

- top_k=None: dùng final_top_k.
- use_graph=None: dùng graph nếu service có graph retriever.
- use_graph=False: ép flat hybrid.
- use_graph=True nhưng service không có graph retriever: code chạy flat branch.
  Muốn bắt buộc graph, phải dựng graph retriever thành công hoặc dùng
  --require-graph ở CLI.

### 8.3. RetrievalResult và EvidenceChain

| Thuộc tính/hàm | Nội dung |
|---|---|
| result.chains | list EvidenceChain chuẩn hóa cho RAG |
| result.evidence | list dict tương thích interface reasoning cũ |
| result.law_evidence | cặp pháp lý đã deduplicate |
| result.trace | backend, reranker, latency, counts và graph statistics |
| result.to_reasoning_payload(...) | JSON-ready payload |
| result.as_dict() | alias tạo payload |

Lưu ý về compatibility: property result.law_evidence là danh sách reference
đã deduplicate theo law_id/aid. Trong payload, key law_evidence hiện giữ các
evidence dict kiểu cũ; dùng evidence_chains nếu consumer cần contract đầy đủ
của EvidenceChain.

Mỗi EvidenceChain có:

~~~text
chain_id, evidence_id, doc_id, content
score, rerank_score, rank
law_evidence, source_chunks, graph_path
provenance, metadata, citation_judgment
~~~

chain.is_graph_backed là True khi graph_path có ít nhất hai node.
provenance có thể chứa retrieval_origin, seed_chunk_id, graph_distance,
cluster_id, cluster_rank và cluster_member_rank.

Dùng trực tiếp cho reasoning:

~~~python
payload = service.retrieve_for_reasoning(
    case_id="case_001",
    query=query,
    case_evidence=["case_chunk_1"],
    top_k=8,
)
# payload gồm query, law_evidence, evidence_chains, retrieval_trace.
~~~

## 16. API cấp thấp

### 9.1. HybridIndexer

~~~python
from retrieval.indexing import HybridIndexer

indexer = HybridIndexer(
    model_name="BAAI/bge-m3",
    use_bge_m3=True,
    use_sentence_transformer=False,
    device="auto",
    batch_size=12,
    max_length=1024,
)
indexer.build_index(documents)
indexer.save_index("data/index")

loaded = HybridIndexer.load_index(
    "data/index",
    model_name="BAAI/bge-m3",
    use_bge_m3=True,
    device="auto",
)
hits = loaded.search(query, top_k=80, alpha=0.50)
~~~

search() trả candidate có doc_id, content, metadata, bm25_score, dense_score
và fused_score. Công thức:

~~~text
fused_score = alpha * normalized_sparse_score
            + (1 - alpha) * dense_cosine_score
~~~

Khi BGE-M3 hoạt động, normalized_sparse_score lấy từ BGE lexical weights;
nếu không có, code dùng rank-bm25 hoặc _SimpleBM25.

### 9.2. Query analysis và routing

~~~python
from retrieval.router import QueryAnalyzer, DocumentRouter

analysis = QueryAnalyzer().analyze(query)
print(analysis.domains)
print(analysis.statute_references)
print(analysis.retrieval_depth)  # fast hoặc deep

routed = DocumentRouter().apply(query, hits)
~~~

Query là deep nếu có statute reference hoặc dài hơn 80 tokens. Router cộng
metadata boost khi law_id hoặc unit_path khớp query.

### 9.3. Reranker

LexicalOverlapReranker cố gắng load BAAI/bge-reranker-v2-m3 qua FlagReranker.
Nếu model/GPU không khả dụng, code fallback deterministic dựa trên base score,
lexical/phrase overlap, citation alignment và legal structure.

~~~python
from retrieval.reranker import LexicalOverlapReranker, ClusterReranker

reranker = LexicalOverlapReranker(
    use_gpu_reranker=True,
    reranker_model_name="BAAI/bge-reranker-v2-m3",
    device="auto",
    batch_size=16,
)
reranked = reranker.rerank(query, routed, top_k=8)
~~~

Graph pipeline mặc định dùng ClusterReranker khi
graph_reranker_method="community_cluster":

~~~python
cluster_reranker = ClusterReranker(
    use_gpu_reranker=True,
    reranker_model_name="BAAI/bge-reranker-v2-m3",
    device="auto",
    batch_size=16,
    cluster_top_k=1,
    cluster_member_top_k=20,
)
~~~

Cluster reranker chấm từng provision, gom theo ontology community trong
graph_path/metadata, rồi tính cluster score:

~~~text
0.65 * max_member_score
+ 0.25 * mean(top-3 member scores)
+ 0.10 * lexical score của cluster summary
~~~

Candidate không có ontology community được giữ như singleton cluster.

### 9.4. Citation usefulness filter

~~~python
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter

citation_filter = HeuristicCitationUsefulnessFilter(
    min_score=0.03,
    max_results=8,
)
filtered = citation_filter.filter(query, reranked)
~~~

Filter kết hợp query overlap, legal-rule signal và retrieval score. Với graph
retrieval, graph-backed evidence có thể giữ lại bằng preserve_graph_paths=True.
