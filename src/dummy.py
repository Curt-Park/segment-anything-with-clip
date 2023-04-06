"""Dummy script for python-project-template."""

import logging
import logging.config

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()


def fibonacci(num: int) -> int:
    """Return fibonacci number."""
    assert num >= 0
    prev, curr = 0, 1
    for _ in range(num):
        curr, prev = curr + prev, curr
    return prev


if __name__ == "__main__":  # pragma: no cover
    logger.critical("Critical Msg")
    logger.error("Error Msg")
    logger.warning("Warning Msg")
    logger.info("Info Msg")
    logger.debug("Debug Msg")
