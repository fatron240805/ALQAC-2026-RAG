"""HTTP client cho ALQAC 2026 Leaderboard API.

SECURITY: secret token đọc từ biến môi trường (`ALQAC_TEAM_TOKEN`), KHÔNG
BAO GIỜ hardcode vào source. Set token bằng 1 trong 2 cách:
  - PowerShell (phiên hiện tại):  $env:ALQAC_TEAM_TOKEN = "alqac_..."
  - File .env (đã gitignore):     copy .env.example -> .env, điền token,
                                   rồi `pip install python-dotenv` và
                                   `from dotenv import load_dotenv; load_dotenv()`
                                   ở đầu run_pipeline.py nếu muốn tự động load.

STATUS: endpoint path/schema bên dưới là PLACEHOLDER — cần cập nhật theo
đúng nội dung https://alqac2026-leaderboard.ngrok.app/api-docs (chưa fetch
được vì đây là ngrok tunnel riêng, không có trong index của web_search).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests  # pip install requests

from orchestration.rate_limiter import RateLimiter


class LeaderboardAuthError(RuntimeError):
    """Thiếu hoặc sai ALQAC_TEAM_TOKEN."""


class LeaderboardSubmissionError(RuntimeError):
    """API trả lỗi khi submit (4xx/5xx)."""


@dataclass
class LeaderboardClient:
    base_url: str
    token: str
    rate_limiter: RateLimiter

    @classmethod
    def from_env(cls, rate_limiter: RateLimiter | None = None) -> "LeaderboardClient":
        token = os.environ.get("ALQAC_TEAM_TOKEN")
        if not token:
            raise LeaderboardAuthError(
                "Thiếu biến môi trường ALQAC_TEAM_TOKEN. Set bằng:\n"
                '  PowerShell: $env:ALQAC_TEAM_TOKEN = "alqac_..."\n'
                "  hoặc copy .env.example -> .env và điền token (KHÔNG commit .env)."
            )
        base_url = os.environ.get(
            "ALQAC_LEADERBOARD_BASE_URL", "https://alqac2026-leaderboard.ngrok.app"
        )
        return cls(
            base_url=base_url.rstrip("/"),
            token=token,
            rate_limiter=rate_limiter or RateLimiter(min_interval_seconds=5.0),
        )

    def _headers(self) -> dict[str, str]:
        # TODO: xác nhận đúng header auth theo api-docs — placeholder dùng
        # Bearer token, một số cuộc thi dùng header riêng (X-Team-Token, ...).
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def submit_predictions(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Gửi file predictions lên leaderboard.

        TODO: cập nhật đúng path + request body schema theo api-docs thật.
        Hiện đang giả định POST /submit với body {"predictions": [...]}.
        """
        self.rate_limiter.wait()  # luôn rate-limit TRƯỚC khi gọi, kể cả lần đầu
        response = requests.post(
            f"{self.base_url}/submit",  # TODO: xác nhận path thật
            json={"predictions": predictions},
            headers=self._headers(),
            timeout=30,
        )
        if response.status_code == 401 or response.status_code == 403:
            raise LeaderboardAuthError(
                f"Token bị từ chối (HTTP {response.status_code}). Kiểm tra lại ALQAC_TEAM_TOKEN."
            )
        if response.status_code == 429:
            raise LeaderboardSubmissionError(
                "HTTP 429 — vượt rate limit (1 req/5s) hoặc quota 3 submit/ngày phía server. "
                "Không retry ngay, đợi server-side quota reset."
            )
        if not response.ok:
            raise LeaderboardSubmissionError(
                f"Submit thất bại: HTTP {response.status_code} — {response.text[:500]}"
            )
        return response.json()
