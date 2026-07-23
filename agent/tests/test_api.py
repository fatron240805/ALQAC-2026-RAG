from pathlib import Path

from fastapi.testclient import TestClient

from app.config import clear_settings_cache
from app.main import app


def test_health(settings, monkeypatch):
    clear_settings_cache()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "error")
    if body["status"] == "ok":
        assert body["config"]["public_case_retrieval_enabled"] is False
        assert body["config"]["official_api_enabled"] is False
        assert "openai_model" in body["config"]


def test_public_test_path_is_local():
    from app.main import DEFAULT_PUBLIC_TEST

    assert isinstance(DEFAULT_PUBLIC_TEST, Path)
    assert DEFAULT_PUBLIC_TEST.name == "ALQAC2026_public_test.json"
    assert "alqac-2026-rag" not in str(DEFAULT_PUBLIC_TEST)


def test_submission_endpoint_rejects_empty(settings, monkeypatch):
    clear_settings_cache()
    client = TestClient(app)
    r = client.post("/v1/submission", json={"cases": []})
    assert r.status_code == 422 or r.status_code == 400


def test_debug_endpoint_requires_matching_id(settings, monkeypatch):
    clear_settings_cache()
    client = TestClient(app)
    r = client.post(
        "/v1/cases/case_1/debug",
        json={"case_id": "case_2", "case_query": "test"},
    )
    assert r.status_code == 400
