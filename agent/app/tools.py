# """Tool adapters with strict Pydantic contracts. Manager-only routing."""

# from __future__ import annotations

# import logging
# from typing import Any

# import httpx

# from app.config import Settings
# from app.observability import Observability
# from app.rag import LawRAG
# from app.schemas import (
#     ElementGraph,
#     LawHit,
#     OfficialCallLedger,
#     OfficialCaseHit,
#     PublicCaseHit,
# )

# logger = logging.getLogger("alqac.tools")


# class PublicCaseSearchTool:
#     """HTTP public judgment search. Disabled => zero network calls."""

#     def __init__(self, settings: Settings, obs: Observability | None = None) -> None:
#         self.settings = settings
#         self.obs = obs

#     def __call__(self, query: str) -> list[PublicCaseHit]:
#         if not self.settings.public_case_retrieval_enabled:
#             logger.info("public_case_search skipped: disabled")
#             return []
#         url = self.settings.public_case_retrieval_url.rstrip("/") + "/search"
#         headers = {
#             "Authorization": f"Bearer {self.settings.public_case_retrieval_api_key}",
#             "Content-Type": "application/json",
#         }
#         with httpx.Client(timeout=30.0) as client:
#             resp = client.post(url, json={"query": query, "top_k": 5}, headers=headers)
#             resp.raise_for_status()
#             data = resp.json()
#         items = data if isinstance(data, list) else data.get("results", [])
#         out: list[PublicCaseHit] = []
#         for item in items:
#             out.append(
#                 PublicCaseHit(
#                     source_id=str(item.get("source_id", item.get("id", ""))),
#                     text=str(item.get("text", item.get("content", ""))),
#                     score=float(item.get("score", 0.0)),
#                 )
#             )
#         return out


# class OfficialCaseTop1Tool:
#     """Official Case Content API — always top-1. Appends chunk_id only."""

#     def __init__(
#         self,
#         settings: Settings,
#         ledger: OfficialCallLedger,
#         obs: Observability | None = None,
#     ) -> None:
#         self.settings = settings
#         self.ledger = ledger
#         self.obs = obs

#     def __call__(self, query: str) -> OfficialCaseHit | None:
#         if not self.settings.official_api_enabled:
#             logger.info("official_case_top1 skipped: disabled")
#             return None
#         if not self.ledger.can_call():
#             logger.info(
#                 "official_case_top1 skipped: ledger stopped (%s)",
#                 self.ledger.stopped_reason,
#             )
#             return None

#         url = self.settings.official_api_url.rstrip("/") + "/search"
#         headers = {
#             "Authorization": f"Bearer {self.settings.official_api_key}",
#             "Content-Type": "application/json",
#         }
#         # Always top-1
#         with httpx.Client(timeout=30.0) as client:
#             resp = client.post(
#                 url,
#                 json={"query": query, "top_k": 1},
#                 headers=headers,
#             )
#             resp.raise_for_status()
#             data = resp.json()

#         hit_raw: dict[str, Any] | None
#         if isinstance(data, list):
#             hit_raw = data[0] if data else None
#         elif isinstance(data, dict):
#             if "chunk_id" in data:
#                 hit_raw = data
#             else:
#                 results = data.get("results") or data.get("items") or []
#                 hit_raw = results[0] if results else None
#         else:
#             hit_raw = None

#         if not hit_raw:
#             self.ledger.record(None, is_duplicate=False, is_no_gain=True)
#             return None

#         chunk_id = str(hit_raw.get("chunk_id", ""))
#         text = str(hit_raw.get("text", hit_raw.get("content", "")))
#         score = hit_raw.get("score")
#         is_dup = bool(chunk_id and chunk_id in self.ledger.seen_chunk_ids)
#         is_no_gain = is_dup or not chunk_id
#         self.ledger.record(chunk_id or None, is_duplicate=is_dup, is_no_gain=is_no_gain)

#         if not chunk_id:
#             return None
#         return OfficialCaseHit(
#             chunk_id=chunk_id,
#             text=text,
#             score=float(score) if score is not None else None,
#         )


# class LawGraphSearchTool:
#     """Local vector + graph. Only source of {law_id, aid} evidence."""

#     def __init__(self, settings: Settings, rag: LawRAG | None = None) -> None:
#         self.settings = settings
#         self.rag = rag or LawRAG(settings)

#     def __call__(
#         self,
#         query: str,
#         element_graph: ElementGraph | None = None,
#     ) -> list[LawHit]:
#         return self.rag.search(query=query, element_graph=element_graph)


"""Tool adapters with strict Pydantic contracts. Manager-only routing."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings
from app.observability import Observability
from app.rag import LawRAG
from app.schemas import (
    ElementGraph,
    LawHit,
    OfficialCallLedger,
    OfficialCaseHit,
    PublicCaseHit,
)

logger = logging.getLogger("alqac.tools")


class PublicCaseSearchTool:
    """HTTP public judgment search. Disabled => zero network calls."""

    def __init__(self, settings: Settings, obs: Observability | None = None) -> None:
        self.settings = settings
        self.obs = obs

    def __call__(self, query: str) -> list[PublicCaseHit]:
        if not self.settings.public_case_retrieval_enabled:
            logger.info("public_case_search skipped: disabled")
            return []
        url = self.settings.public_case_retrieval_url.rstrip("/") + "/search"
        headers = {
            "Authorization": f"Bearer {self.settings.public_case_retrieval_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"query": query, "top_k": 5}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = data if isinstance(data, list) else data.get("results", [])
        out: list[PublicCaseHit] = []
        for item in items:
            out.append(
                PublicCaseHit(
                    source_id=str(item.get("source_id", item.get("id", ""))),
                    text=str(item.get("text", item.get("content", ""))),
                    score=float(item.get("score", 0.0)),
                )
            )
        return out


class OfficialCaseTop1Tool:
    """Official Case Content API — always top-1. Appends chunk_id only.

    Confirmed contract (from the organizers' official API docs at
    alqac-api.ngrok.pro): POST /retrieve, header X-API-Key, body
    {"query": ..., "case_id": ...}. Response shape:
    {"results": [{"score": ..., "text": ..., "chunk_id": ...}]}.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: OfficialCallLedger,
        obs: Observability | None = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.obs = obs

    def __call__(self, query: str, case_id: str) -> OfficialCaseHit | None:
        if not self.settings.official_api_enabled:
            logger.info("official_case_top1 skipped: disabled")
            return None
        if not self.ledger.can_call():
            logger.info(
                "official_case_top1 skipped: ledger stopped (%s)",
                self.ledger.stopped_reason,
            )
            return None

        url = self.settings.official_api_url.rstrip("/") + "/retrieve"
        headers = {
            "X-API-Key": self.settings.official_api_key,
            "Content-Type": "application/json",
        }
        # Always top-1
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url,
                json={"query": query, "case_id": case_id},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        hit_raw: dict[str, Any] | None
        if isinstance(data, list):
            hit_raw = data[0] if data else None
        elif isinstance(data, dict):
            if "chunk_id" in data:
                hit_raw = data
            else:
                results = data.get("results") or data.get("items") or []
                hit_raw = results[0] if results else None
        else:
            hit_raw = None

        if not hit_raw:
            self.ledger.record(None, is_duplicate=False, is_no_gain=True)
            return None

        chunk_id = str(hit_raw.get("chunk_id", ""))
        text = str(hit_raw.get("text", hit_raw.get("content", "")))
        score = hit_raw.get("score")
        is_dup = bool(chunk_id and chunk_id in self.ledger.seen_chunk_ids)
        is_no_gain = is_dup or not chunk_id
        self.ledger.record(chunk_id or None, is_duplicate=is_dup, is_no_gain=is_no_gain)

        if not chunk_id:
            return None
        return OfficialCaseHit(
            chunk_id=chunk_id,
            text=text,
            score=float(score) if score is not None else None,
        )


class LawGraphSearchTool:
    """Local vector + graph. Only source of {law_id, aid} evidence."""

    def __init__(self, settings: Settings, rag: LawRAG | None = None) -> None:
        self.settings = settings
        self.rag = rag or LawRAG(settings)

    def __call__(
        self,
        query: str,
        element_graph: ElementGraph | None = None,
    ) -> list[LawHit]:
        return self.rag.search(query=query, element_graph=element_graph)