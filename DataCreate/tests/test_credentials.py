from datacreate import credentials
from datacreate.feedback import _secret


def test_saved_user_key_works_without_inherited_variable(monkeypatch):
    monkeypatch.delenv("SSSTOKEN_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_windows_user_variable", lambda name: "saved-test-key")
    assert credentials.credential_value("SSSTOKEN_API_KEY") == "saved-test-key"
    assert _secret("SSSTOKEN_API_KEY") == "saved-test-key"


def test_process_override_and_explicit_empty_value_take_precedence(monkeypatch):
    monkeypatch.setattr(credentials, "_windows_user_variable", lambda name: "saved-test-key")
    monkeypatch.setenv("SSSTOKEN_API_KEY", " process-test-key ")
    assert credentials.credential_value("SSSTOKEN_API_KEY") == "process-test-key"
    monkeypatch.setenv("SSSTOKEN_API_KEY", "")
    assert credentials.credential_value("SSSTOKEN_API_KEY") == ""


def test_unrelated_keys_never_read_from_registry(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    def unexpected(name):
        raise AssertionError("Unrelated key read from registry")
    monkeypatch.setattr(credentials, "_windows_user_variable", unexpected)
    assert credentials.credential_value("OPENAI_API_KEY") == ""
