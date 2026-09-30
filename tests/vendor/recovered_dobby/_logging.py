"""Logger configuration.

Usage at application layer:
    # Option 1: Environment variable (recommended)
    export DOBBY_LOG=DEBUG  # or INFO, WARNING, ERROR

    # Option 2: Configure logger in your app
    import logging
    logging.getLogger("dobby").setLevel(logging.DEBUG)
    logging.getLogger("dobby").addHandler(logging.StreamHandler())

Why silent by default?
    - Avoids log spam in production applications
    - Lets the application control all logging configuration
    - Prevents conflicts with application's logging setup
"""

import logging
import os

logger = logging.getLogger("dobby")

_level = os.getenv("DOBBY_LOG", "").upper()
if _level in ("DEBUG", "INFO", "WARNING", "ERROR"):
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, _level))

__all__ = ["logger"]
