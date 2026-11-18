from tp2te.core.config import load_config, load_dotenv

# conftest.isolated_llm_env clears the LLM variables before each test.


def test_dotenv_fills_llm_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nLLM_BASE_URL=https://openrouter.ai/api/v1\n'
                   'export OPENROUTER_API_KEY="sk-test"\nMODEL=some/model  # note\n\n')
    for n in ("LLM_BASE_URL", "OPENROUTER_API_KEY", "MODEL"):
        monkeypatch.delenv(n, raising=False)   # registered so monkeypatch removes them afterwards
    assert set(load_dotenv(env)) == {"LLM_BASE_URL", "OPENROUTER_API_KEY", "MODEL"}
    llm = load_config(None, env=True).llm
    assert (llm.base_url, llm.model, llm.api_key_env) == ("https://openrouter.ai/api/v1", "some/model",
                                                          "OPENROUTER_API_KEY")


def test_shell_env_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL", "from-shell")
    (tmp_path / ".env").write_text("MODEL=from-file\n")
    assert load_dotenv(tmp_path / ".env") == []
    assert load_config(None, env=True).llm.model == "from-shell"


def test_toml_wins_and_placeholders_are_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://env")
    monkeypatch.setenv("MODEL", "env-model")
    monkeypatch.setenv("TP2TE_API_KEY", "k")
    toml = tmp_path / "c.toml"
    toml.write_text('[llm]\nbase_url = "https://..."\nmodel = "toml-model"\napi_key_env = "MY_KEY"\n')
    llm = load_config(toml, env=True).llm
    assert (llm.base_url, llm.model, llm.api_key_env) == ("https://env", "toml-model", "MY_KEY")


def test_no_env_by_default(monkeypatch):
    monkeypatch.setenv("MODEL", "env-model")
    assert load_config(None).llm.model == ""


def test_missing_dotenv_is_fine(tmp_path):
    assert load_dotenv(tmp_path / ".env") == []
