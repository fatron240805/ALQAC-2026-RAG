from app.config import Settings, clear_settings_cache


def test_defaults_disable_retrieval_agents(settings):
    assert settings.public_case_retrieval_enabled is False
    assert settings.official_api_enabled is False
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.manager_max_iterations == 5
    assert settings.official_call_budget_multiplier == 2.0


def test_official_budget(settings):
    assert settings.official_max_calls(10) == 20


def test_missing_chat_config_fails(monkeypatch):
    clear_settings_cache()
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    try:
        Settings()
        assert False, "expected ValueError"
    except Exception as exc:
        assert "OPENAI_BASE_URL" in str(exc)


def test_enabled_official_requires_credentials(monkeypatch, tmp_path):
    clear_settings_cache()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setenv("OFFICIAL_API_ENABLED", "true")
    monkeypatch.setenv("OFFICIAL_API_URL", "")
    monkeypatch.setenv("OFFICIAL_API_KEY", "")
    try:
        Settings()
        assert False, "expected ValueError"
    except Exception as exc:
        assert "OFFICIAL_API" in str(exc)


def test_budget_multiplier_range(monkeypatch):
    clear_settings_cache()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setenv("OFFICIAL_CALL_BUDGET_MULTIPLIER", "3")
    try:
        Settings()
        assert False, "expected ValueError"
    except Exception as exc:
        assert "OFFICIAL_CALL_BUDGET_MULTIPLIER" in str(exc)
