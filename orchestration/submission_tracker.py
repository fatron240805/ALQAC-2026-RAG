"""Leaderboard submission guardrails: max N/day, logged with config hash.

Per Plan.md: "Build submission guardrails: max 3 submissions/day; queue API
calls at 5-second intervals; save every submitted version with config hash."
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path


class SubmissionGuardrailError(RuntimeError):
    """Raised when a submission would exceed the daily quota."""


class SubmissionTracker:
    _FIELDS = ["timestamp", "date", "config_hash", "public_score", "notes"]

    def __init__(self, csv_path: str | Path, max_per_day: int = 3):
        self.csv_path = Path(csv_path)
        self.max_per_day = max_per_day
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._FIELDS).writeheader()

    def _count_for_date(self, date_str: str) -> int:
        count = 0
        with self.csv_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("date") == date_str:
                    count += 1
        return count

    def remaining_today(self) -> int:
        today = dt.date.today().isoformat()
        return max(0, self.max_per_day - self._count_for_date(today))

    def register_submission(
        self, config_hash: str, public_score: float | None = None, notes: str = ""
    ) -> None:
        now = dt.datetime.now()
        today = now.date().isoformat()
        if self._count_for_date(today) >= self.max_per_day:
            raise SubmissionGuardrailError(
                f"Đã đạt giới hạn {self.max_per_day} lần submit/ngày ({today}). "
                "Dừng lại để không vi phạm luật cuộc thi."
            )
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._FIELDS)
            writer.writerow(
                {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "date": today,
                    "config_hash": config_hash,
                    "public_score": "" if public_score is None else public_score,
                    "notes": notes,
                }
            )
