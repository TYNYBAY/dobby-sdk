"""Loguru logging configuration for dobby SDK.

Configuration via environment variables:
    DOBBY_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    DOBBY_LOG_FORMAT: "pretty" or "json" (default: pretty)

Example:
    # Enable debug logging
    import os
    os.environ["DOBBY_LOG_LEVEL"] = "DEBUG"

    from dobby._logging import logger
    logger.info("Hello from dobby!")
"""

import os
import sys

from loguru import logger

# Remove default handler to configure our own
logger.remove()

LOG_LEVEL = os.getenv("DOBBY_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("DOBBY_LOG_FORMAT", "pretty").lower()

if LOG_FORMAT == "json":
    # JSON format for log aggregation systems (production)
    logger.add(
        sys.stderr,
        format="{message}",
        level=LOG_LEVEL,
        serialize=True,  # Loguru's built-in JSON serialization
    )
else:
    # Pretty colored format for development
    # Loguru auto-detects TTY and enables/disables colors accordingly (True if TTY, False otherwise)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=LOG_LEVEL,
        colorize=None,
        backtrace=True,
        diagnose=True,
    )

__all__ = ["logger"]
