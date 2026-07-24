# Changelog

Append-only log. Format: `## [YYYY-MM-DD] type | description`

## [2026-07-24] reliability | Incremental submission checkpoints

- Atomically write a case-specific artifact after each completed case
- Preserve ALQAC-valid rows in `submission_<case_id>.json` and persist rejected/error case details in `error_<case_id>.json`

## [2026-07-24] reliability | Per-case artifact filenames

- Save valid case outputs as `submission_<case_id>.json` and failed cases as `error_<case_id>.json`
- Return plural artifact paths from submission endpoints for batch requests

## [2026-07-24] reliability | Structured-output validation recovery

- Normalize weak-model list/object variants in agent result schemas, including format-review issues
- Retry every agent role after JSON parse or Pydantic validation failures; exhausted failures remain isolated to saved error cases

## [2026-07-22] init | Wiki created

- Created AGENTS.md with codebase conventions
- Created wiki/ with 10 entity/concept pages
- Indexed all app/*.py modules
- Documented provenance rules, retrieval, security

## [2026-07-22] security | Three findings fixed

- Atomic index rebuild with alias swap (no delete-first)
- Public-test endpoint: removed caller-controlled path
- Concurrent submissions: request-scoped output + atomic rename

## [2026-07-22] security | Second round fixes

- Auth + batch-size cap + rate limit on execution endpoints
- Paginated scroll in index build (no 10k truncation)
- Minimum-evidence enforcement: law_evidence always required

## [2026-07-23] reliability | Model API retry

- Reused transient OpenAI error predicate for agent calls
- Retry model invocation up to 5 attempts with 1s exponential backoff
- JSON/schema failures remain fail-fast

## [2026-07-23] reliability | Wrapped model-server retry

- Retry LangChain `ValueError` responses containing rate-limit, 408, 429, or 5xx upstream failures
- Use six attempts with randomized 2–60s exponential backoff; disable nested client retries
- Preserve fail-fast behavior for JSON and schema errors

## [2026-07-23] prompts | Simplified agent instructions

- Removed paper, PDF, local-retrieval, and generated-instruction artifacts from system prompts
- Kept Vietnamese role instructions while retaining English JSON keys and route values as machine contracts
- Made law-query role return plain text instead of conflicting JSON-only output

## [2026-07-23] prompts | Restored role responsibilities

- Restored the Table 8 responsibilities for all six roles as Vietnamese instructions
- Retained only ALQAC output contracts and evidence constraints; no paper-path or retrieval-implementation text appears in system prompts

## [2026-07-23] observability | Raw agent prompt debug logging

- Added `configure_debug_file_handler()` and `log_raw_prompt()` in `app/observability.py`
- Per-process timestamped log file at `artifacts/logs/agent-prompts-YYYYMMDDTHHMMSS±ZZZZ.log`
- `_invoke_role()` logs exact system + user prompt before each model invocation
- Removed `redact()` from `log_event`, Langfuse spans (input/metadata/output), and debug-state helper
- Debug endpoint returns unredacted state
- 10 new tests in `tests/test_debug_logging.py` verifying file creation, content, unredacted paths

## [2026-07-23] ingest | Private-test law corpus indexed

- Flattened nested `content` article documents while preserving `law_id` and `aid`
- Built and atomically activated 2,820 private-test law vectors across 14 laws
- Graph rebuilt with 2,820 nodes and 2,806 adjacency edges
- Embedding failures retry and resumable temporary collections preserve completed points
- Qdrant alias and `query_points` compatibility verified against local persistent storage
