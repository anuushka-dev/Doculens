import logging

import pytest

from app.core.logging import (
    LOG_FORMAT,
    LOGGER_NAME,
    _DocuLensStreamHandler,
    configure_logging,
)


@pytest.fixture
def isolated_app_logger():
    """Restore logger state after each test."""
    logger = logging.getLogger(LOGGER_NAME)

    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate

    for handler in original_handlers:
        logger.removeHandler(handler)

    try:
        yield logger
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

            if handler not in original_handlers:
                handler.close()

        for handler in original_handlers:
            logger.addHandler(handler)

        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_normalizes_logging_level(
    isolated_app_logger: logging.Logger,
) -> None:
    configure_logging(" debug ")

    assert isolated_app_logger.level == logging.DEBUG


def test_rejects_unsupported_level_without_mutation(
    isolated_app_logger: logging.Logger,
) -> None:
    original_level = isolated_app_logger.level
    original_handlers = isolated_app_logger.handlers[:]
    original_propagate = isolated_app_logger.propagate

    with pytest.raises(ValueError, match="Unsupported logging level"):
        configure_logging("TRACE")

    assert isolated_app_logger.level == original_level
    assert isolated_app_logger.handlers == original_handlers
    assert isolated_app_logger.propagate == original_propagate


def test_rejects_non_string_level(
    isolated_app_logger: logging.Logger,
) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        configure_logging(None)  # type: ignore[arg-type]


def test_repeated_configuration_does_not_duplicate_handlers(
    isolated_app_logger: logging.Logger,
) -> None:
    configure_logging("INFO")
    initial_handlers = isolated_app_logger.handlers[:]

    configure_logging("DEBUG")

    assert isolated_app_logger.handlers == initial_handlers
    assert sum(
        isinstance(handler, _DocuLensStreamHandler)
        for handler in isolated_app_logger.handlers
    ) == 1


def test_does_not_modify_root_logger(
    isolated_app_logger: logging.Logger,
) -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]

    configure_logging("WARNING")

    assert root_logger.level == original_level
    assert root_logger.handlers == original_handlers


def test_applies_expected_formatter(
    isolated_app_logger: logging.Logger,
) -> None:
    configure_logging("INFO")

    handler = next(
        handler
        for handler in isolated_app_logger.handlers
        if isinstance(handler, _DocuLensStreamHandler)
    )

    assert handler.formatter is not None
    assert handler.formatter._fmt == LOG_FORMAT


def test_preserves_externally_managed_handler(
    isolated_app_logger: logging.Logger,
) -> None:
    external_handler = logging.NullHandler()
    isolated_app_logger.addHandler(external_handler)

    configure_logging("INFO")

    assert external_handler in isolated_app_logger.handlers
    assert not any(
        isinstance(handler, _DocuLensStreamHandler)
        for handler in isolated_app_logger.handlers
    )