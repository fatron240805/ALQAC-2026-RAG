"""HTTP client cho ALQAC 2026 Retrieval API (case-content evidence).

CONFIRMED từ api-docs (https://alqac2026-leaderboard.ngrok.app/api-docs):
  - Base URL: https://alqac-api.ngrok.pro  (KHÁC domain với leaderboard)
  - Auth header: X-API-Key: <token>
  - POST /retrieve {query, case_id} -> {"results": [{"score","text","chunk_id"}]}
  - Trả về đúng 1 segment/call — muốn nhiều evidence phải gọi nhiều lần,
    mỗi lần tính vào c_i (đo từ log server, KHÔNG submit c_i thủ công).
  - Rate limit: 1 request / 5s / team — áp dụng cho ĐÚNG endpoint này.

Công thức điểm liên quan (từ api-docs, khớp evaluation/metrics.py):
    E_i = max(0, 1 - max(0, c_i - 2*n_i) / (3*n_i))
    (không phạt tới 2*n_i calls, giảm dần về 0 tại 5*n_i calls)

CHƯA CONFIRM: docs không cho cách biết trước n_i (số segment/case) từ
phía client -> chiến lược số lần gọi/case hiện là heuristic
(config.max_case_retrieval_calls), không tối ưu theo n_i thật. Nếu API
có endpoint lộ n_i (vd GET /case/{case_id}/info), cập nhật lại đây.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

from orchestration.rate_limiter import RateLimiter


class CaseRetrievalAuthError(RuntimeError):
    """X-API-Key bị từ chối (403)."""


class CaseRetrievalError(RuntimeError):
    """Lỗi khác từ API (422 / 429 / 503)."""


@dataclass(frozen=True)
class CaseEvidenceHit:
    chunk_id: str
    text: str
    score: float


@dataclass
class CaseRetrievalClient:
    base_url: str
    token: str
    rate_limiter: RateLimiter
    call_count: int = field(default=0, init=False)

    @classmethod
    def from_env(cls, rate_limiter: RateLimiter | None = None) -> "CaseRetrievalClient":
        token = os.environ.get("ALQAC_TEAM_TOKEN")
        if not token:
            raise CaseRetrievalAuthError(
                "Thiếu biến môi trường ALQAC_TEAM_TOKEN. Xem .env.example."
            )
        base_url = os.environ.get("ALQAC_RETRIEVAL_API_BASE_URL", "https://alqac-api.ngrok.pro")
        return cls(
            base_url=base_url.rstrip("/"),
            token=token,
            rate_limiter=rate_limiter or RateLimiter(min_interval_seconds=5.0),
        )

    def retrieve(self, query: str, case_id: str) -> CaseEvidenceHit:
        """1 lần gọi = 1 segment top-1. LUÔN rate-limit trước khi gọi (kể cả lần đầu)."""
        self.rate_limiter.wait()
        response = requests.post(
            f"{self.base_url}/retrieve",
            headers={"X-API-Key": self.token, "Content-Type": "application/json"},
            json={"query": query, "case_id": case_id},
            timeout=30,
        )
        self.call_count += 1

        if response.status_code == 403:
            raise CaseRetrievalAuthError("X-API-Key bị từ chối (403). Kiểm tra lại ALQAC_TEAM_TOKEN.")
        if response.status_code == 422:
            raise CaseRetrievalError(
                f"Request sai định dạng (422) — thiếu query/case_id? body={response.text[:300]}"
            )
        if response.status_code == 429:
            raise CaseRetrievalError(
                "429 — vượt rate limit dù đã có RateLimiter. Kiểm tra có tiến trình khác "
                "cùng dùng chung token/rate limiter không (vd chạy song song 2 process)."
            )
        if response.status_code == 503:
            raise CaseRetrievalError("503 — team database tạm thời không sẵn sàng, thử lại sau.")
        response.raise_for_status()

        results = response.json().get("results", [])
        if not results:
            return CaseEvidenceHit(chunk_id="", text="", score=0.0)
        top = results[0]
        return CaseEvidenceHit(chunk_id=top["chunk_id"], text=top["text"], score=top["score"])

    def retrieve_multi(self, queries: list[str], case_id: str) -> list[CaseEvidenceHit]:
        """Gọi tuần tự nhiều query cho cùng 1 case. Mỗi query = 1 call = tính vào c_i,
        nên chỉ nên gọi khi thực sự cần thêm evidence (xem docstring module về E_i).
        De-dup theo chunk_id để tránh evidence trùng lặp không cần thiết.
        """
        hits: list[CaseEvidenceHit] = []
        seen: set[str] = set()
        for query in queries:
            hit = self.retrieve(query, case_id)
            if hit.chunk_id and hit.chunk_id not in seen:
                hits.append(hit)
                seen.add(hit.chunk_id)
        return hits
