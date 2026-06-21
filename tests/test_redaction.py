from __future__ import annotations

from pathlib import Path

import pytest

from engine.core.redaction import (
    redact_text,
    safe_config_label,
    safe_path,
    sanitize_exception,
)


def test_redact_text_masks_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_token_value")
    message = "Request failed with authorization hf_super_secret_token_value"
    assert "hf_super_secret_token_value" not in redact_text(message)
    assert "***" in redact_text(message)


def test_redact_text_masks_key_value_patterns() -> None:
    text = "Invalid credentials: key: abc12345 and token=deadbeef"
    redacted = redact_text(text)
    assert "abc12345" not in redacted
    assert "deadbeef" not in redacted


def test_redact_text_masks_sas_signature() -> None:
    url = "https://example.blob.core.windows.net/x?sig=verysecretvalue&se=2030"
    redacted = redact_text(url)
    assert "verysecretvalue" not in redacted
    assert "sig=***" in redacted


def test_safe_path_hides_home_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("engine.core.redaction.Path.home", lambda: tmp_path)
    nested = tmp_path / "fetched" / "era5"
    nested.mkdir(parents=True)
    assert safe_path(nested) == "~/fetched/era5"


def test_safe_config_label_never_includes_username() -> None:
    label = safe_config_label(Path("/home/alice/.cdsapirc"))
    assert label == "~/.cdsapirc"
    assert "alice" not in label


def test_sanitize_exception_applies_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_TOKEN", "foundry-secret-token")
    exc = RuntimeError("Auth failed for foundry-secret-token")
    sanitized = sanitize_exception(exc)
    assert "foundry-secret-token" not in sanitized


def test_cds_config_error_uses_safe_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.ingress.download.cds import CdsConfigError, cds_client

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("engine.ingress.download.paths.user_home", lambda: fake_home)

    with pytest.raises(CdsConfigError, match=r"~/.cdsapirc") as exc:
        cds_client()
    assert str(fake_home) not in str(exc.value)
