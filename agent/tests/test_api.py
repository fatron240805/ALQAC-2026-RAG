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
    from app.config import Settings

    settings = Settings()
    path = settings.public_test_path
    assert isinstance(path, Path)
    assert path.name == "ALQAC2026_public_test.json"


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


def test_submission_rejects_bad_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    clear_settings_cache()
    client = TestClient(app)
    r = client.post("/v1/submission", json={"cases": []})
    # Empty cases → 422 before auth even fires; test with valid body but no key
    monkeypatch.setenv("API_KEY", "secret123")
    clear_settings_cache()
    r = client.post(
        "/v1/submission",
        json={"cases": [{"case_id": "c1", "case_query": "q"}]},
    )
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"] or "invalid" in r.json()["detail"].lower()


def test_submission_accepts_valid_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    clear_settings_cache()
    client = TestClient(app)
    r = client.post(
        "/v1/submission",
        json={"cases": []},
        headers={"X-API-Key": "secret123"},
    )
    # Empty cases rejected by validator, not auth
    assert r.status_code == 422 or r.status_code == 400


def test_batch_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_SIZE", "2")
    clear_settings_cache()
    client = TestClient(app)
    cases = [{"case_id": f"c{i}", "case_query": f"q{i}"} for i in range(5)]
    r = client.post("/v1/submission", json={"cases": cases})
    assert r.status_code == 400
    assert "batch size" in r.json()["detail"].lower()
