import pytest
from pydantic import ValidationError

from app.core.config import Settings

CONFIG_ENV_VARS = (
    "DOCULENS_APP_NAME",
    "DOCULENS_ENVIRONMENT",
    "DOCULENS_LOG_LEVEL",
    "DOCULENS_MAX_UPLOAD_BYTES",
    "DOCULENS_MAX_PDF_PAGES",
)


@pytest.fixture(autouse=True)
def clear_config_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent the developer's environment from leaking into tests."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_settings_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "DocuLens"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.max_pdf_pages == 250


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCULENS_ENVIRONMENT", " TEST ")
    monkeypatch.setenv("DOCULENS_LOG_LEVEL", " debug ")
    monkeypatch.setenv("DOCULENS_MAX_PDF_PAGES", "100")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
    assert settings.max_pdf_pages == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"app_name": ""},
        {"environment": "staging"},
        {"log_level": "TRACE"},
        {"max_upload_bytes": 0},
        {"max_upload_bytes": -1},
        {"max_upload_bytes": 100 * 1024 * 1024 + 1},
        {"max_pdf_pages": 0},
        {"max_pdf_pages": -1},
        {"max_pdf_pages": 5001},
    ],
)
def test_invalid_settings_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_settings_are_immutable() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"  # type: ignore[misc]