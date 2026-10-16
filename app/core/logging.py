import logging
from typing import Final

LOGGER_NAME: Final[str] = "app"

LOG_LEVELS: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"


class _DocuLensStreamHandler(logging.StreamHandler):
    """Identify stream handlers owned by DocuLens."""


def _resolve_level(level: str) -> int:
    """Validate and normalize a logging level."""
    if not isinstance(level, str):
        raise TypeError("Logging level must be a string.")

    normalized = level.strip().upper()

    if not normalized or normalized not in LOG_LEVELS:
        supported = ", ".join(LOG_LEVELS)
        raise ValueError(
            f"Unsupported logging level {level!r}. "
            f"Expected one of: {supported}."
        )

    return LOG_LEVELS[normalized]


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the DocuLens application namespace.

    This function:
    - Validates the level before changing logging state.
    - Does not modify the root logger.
    - Does not replace externally managed handlers.
    - Reuses its own handler when called repeatedly.

    Callers must never include document contents, credentials,
    API keys, or raw prompts in log messages.
    """
    resolved_level = _resolve_level(level)

    app_logger = logging.getLogger(LOGGER_NAME)
    app_logger.setLevel(resolved_level)
    app_logger.propagate = False

    managed_handlers = [
        handler
        for handler in app_logger.handlers
        if isinstance(handler, _DocuLensStreamHandler)
    ]

    if managed_handlers:
        handler = managed_handlers[0]

        # Remove any duplicate handlers owned by DocuLens.
        for duplicate in managed_handlers[1:]:
            app_logger.removeHandler(duplicate)
            duplicate.close()

    elif not app_logger.handlers:
        handler = _DocuLensStreamHandler()
        app_logger.addHandler(handler)

    else:
        # An external handler already owns this logger's output.
        # Preserve it instead of installing a competing handler.
        return

    handler.setLevel(logging.NOTSET)
    handler.setFormatter(
        logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    )