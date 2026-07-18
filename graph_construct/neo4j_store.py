from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ALLOWED_RELATION_TYPES = (
    "CONTAINS",
    "CITES",
    "GOVERNED_BY",
    "RELATED_TO",
    "EXCEPTION_TO",
    "ANALOGOUS_TO",
    "MATCHES_FACT_PATTERN",
)
ALLOWED_URI_SCHEMES = {"neo4j", "neo4j+s", "neo4j+ssc", "bolt", "bolt+s", "bolt+ssc"}


def _configured_env_value(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    return bool(normalized) and not any(
        marker in normalized for marker in ("replace_with", "<", "your_", "chang_me")
    )


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _positive_float_from_env(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str | None = None
    max_connection_pool_size: int = 8
    connection_acquisition_timeout: float = 30.0

    @property
    def is_aura(self) -> bool:
        return urlparse(self.uri).scheme == "neo4j+s"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        uri = os.environ.get("NEO4J_URI")
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        database = str(os.environ.get("NEO4J_DATABASE", "")).strip() or None
        missing = [
            name
            for name, value in (("NEO4J_URI", uri), ("NEO4J_USERNAME", username), ("NEO4J_PASSWORD", password))
            if not _configured_env_value(value)
        ]
        if missing:
            raise RuntimeError(f"Missing or placeholder Neo4j env vars: {', '.join(missing)}")

        scheme = urlparse(str(uri)).scheme
        if scheme not in ALLOWED_URI_SCHEMES:
            allowed = ", ".join(sorted(ALLOWED_URI_SCHEMES))
            raise RuntimeError(f"Unsupported NEO4J_URI scheme '{scheme}'. Use one of: {allowed}.")
        return cls(
            uri=str(uri),
            username=str(username),
            password=str(password),
            database=database,
            max_connection_pool_size=_positive_int_from_env("NEO4J_MAX_CONNECTION_POOL_SIZE", 8),
            connection_acquisition_timeout=_positive_float_from_env(
                "NEO4J_CONNECTION_ACQUISITION_TIMEOUT", 30.0
            ),
        )


class Neo4jGraphStore:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or Neo4jConfig.from_env()
        self._available_relation_types: tuple[str, ...] | None = None
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "neo4j package is not installed. Install it to use the Neo4j graph backend."
            ) from exc
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
            max_connection_pool_size=self.config.max_connection_pool_size,
            connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            telemetry_disabled=True,
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def count_legal_nodes(self) -> int:
        with self._driver.session(database=self.config.database) as session:
            record = session.run("MATCH (n:LegalNode) RETURN count(n) AS count").single()
            if record is None:
                return 0
            return int(record["count"] or 0)

    def available_relation_types(self) -> tuple[str, ...]:
        if self._available_relation_types is not None:
            return self._available_relation_types

        with self._driver.session(database=self.config.database) as session:
            result = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            )
            found = {str(record["relationshipType"]).upper() for record in result}
        self._available_relation_types = tuple(
            relation_type for relation_type in ALLOWED_RELATION_TYPES if relation_type in found
        )
        return self._available_relation_types

    def ensure_schema(self) -> None:
        cypher_statements = [
            "CREATE CONSTRAINT legalnode_node_id IF NOT EXISTS FOR (n:LegalNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT law_law_id IF NOT EXISTS FOR (n:Law) REQUIRE n.law_id IS UNIQUE",
            "CREATE INDEX article_lookup IF NOT EXISTS FOR (n:Article) ON (n.law_id, n.aid)",
            "CREATE INDEX article_number_lookup IF NOT EXISTS FOR (n:Article) ON (n.law_id, n.article_number)",
            "CREATE INDEX concept_alias IF NOT EXISTS FOR (n:Concept) ON (n.normalized_alias)",
            "CREATE INDEX sourcechunk_chunk_id IF NOT EXISTS FOR (n:SourceChunk) ON (n.chunk_id)",
        ]
        with self._driver.session(database=self.config.database) as session:
            for statement in cypher_statements:
                session.run(statement)

    def clear(self) -> None:
        with self._driver.session(database=self.config.database) as session:
            session.run("MATCH (n:LegalNode) DETACH DELETE n")

    def import_graph(
        self,
        node_rows: list[dict[str, Any]],
        edge_rows: list[dict[str, Any]],
        *,
        clear_first: bool = False,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        self.ensure_schema()
        if clear_first:
            self.clear()

        node_count = self._upsert_nodes(node_rows, batch_size=batch_size)
        edge_count = self._upsert_edges(edge_rows, batch_size=batch_size)
        return {"nodes_upserted": node_count, "edges_upserted": edge_count}

    def _upsert_nodes(self, node_rows: list[dict[str, Any]], *, batch_size: int) -> int:
        if not node_rows:
            return 0

        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in node_rows:
            labels = tuple(sorted(label for label in row.get("labels", []) if label and label != "LegalNode"))
            grouped.setdefault(labels, []).append(row)

        total = 0
        with self._driver.session(database=self.config.database) as session:
            for labels, rows in grouped.items():
                label_clause = "".join(f":{label}" for label in labels)
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MERGE (n:LegalNode {{node_id: row.node_id}}) "
                    f"SET n += row.properties "
                    f"SET n{label_clause}"
                )
                for batch in _batched(rows, batch_size):
                    session.run(cypher, rows=batch)
                    total += len(batch)
        return total

    def _upsert_edges(self, edge_rows: list[dict[str, Any]], *, batch_size: int) -> int:
        if not edge_rows:
            return 0

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in edge_rows:
            edge_type = str(row.get("edge_type") or "").strip().upper()
            if edge_type not in ALLOWED_RELATION_TYPES:
                continue
            grouped.setdefault(edge_type, []).append(row)

        total = 0
        with self._driver.session(database=self.config.database) as session:
            for edge_type, rows in grouped.items():
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MATCH (src:LegalNode {{node_id: row.src}}) "
                    f"MATCH (dst:LegalNode {{node_id: row.dst}}) "
                    f"MERGE (src)-[r:{edge_type}]->(dst) "
                    f"SET r += row.properties"
                )
                for batch in _batched(rows, batch_size):
                    session.run(cypher, rows=batch)
                    total += len(batch)
        return total

    def seed_nodes_for_chunk(self, chunk_id: str) -> list[str]:
        query = (
            "MATCH (n:LegalNode) "
            "WHERE n.source_chunk_id = $chunk_id OR n.chunk_id = $chunk_id "
            "RETURN n.node_id AS node_id "
            "ORDER BY CASE WHEN n.node_type = 'article' THEN 0 ELSE 1 END, n.node_id"
        )
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, chunk_id=chunk_id)
            return [str(record["node_id"]) for record in result if record.get("node_id")]

    def expand_from_seeds(
        self,
        seed_node_ids: Iterable[str],
        *,
        depth: int = 1,
        target_layers: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        seed_ids = [str(seed_id) for seed_id in seed_node_ids if str(seed_id).strip()]
        if not seed_ids:
            return []

        relation_types = self.available_relation_types()
        if not relation_types:
            return []
        relation_pattern = "|".join(relation_types)
        bounded_depth = max(1, min(int(depth), 5))
        cypher = (
            f"UNWIND $seed_ids AS seed_id "
            f"MATCH (seed:LegalNode {{node_id: seed_id}}) "
            f"MATCH path = (seed)-[:{relation_pattern}*1..{bounded_depth}]-(candidate:LegalNode) "
            f"WHERE $target_layers IS NULL OR candidate.layer IN $target_layers "
            f"RETURN candidate.node_id AS node_id, "
            f"labels(candidate) AS labels, "
            f"candidate.layer AS layer, "
            f"candidate.node_type AS node_type, "
            f"candidate.law_id AS law_id, "
            f"candidate.aid AS aid, "
            f"candidate.source_aid AS source_aid, "
            f"candidate.article_number AS article_number, "
            f"candidate.article_index AS article_index, "
            f"candidate.source_chunk_id AS source_chunk_id, "
            f"candidate.chunk_id AS chunk_id, "
            f"candidate.text AS text, "
            f"candidate.unit_path AS unit_path, "
            f"candidate.aliases AS aliases, "
            f"candidate.normalized_alias AS normalized_alias, "
            f"length(path) AS distance, "
            f"[n IN nodes(path) | n.node_id] AS graph_path "
            f"ORDER BY distance ASC, node_id ASC "
            f"LIMIT $limit"
        )
        target_layers_list = list(target_layers) if target_layers is not None else None
        with self._driver.session(database=self.config.database) as session:
            result = session.run(
                cypher,
                seed_ids=seed_ids,
                target_layers=target_layers_list,
                limit=int(limit),
            )
            rows = [dict(record) for record in result]
        return _dedupe_by_best_distance(rows)


def _dedupe_by_best_distance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            continue
        current_distance = int(row.get("distance") or 0)
        previous = best.get(node_id)
        if previous is None or current_distance < int(previous.get("distance") or 0):
            best[node_id] = row
    return sorted(best.values(), key=lambda item: (int(item.get("distance") or 0), str(item.get("node_id") or "")))[:]


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]
