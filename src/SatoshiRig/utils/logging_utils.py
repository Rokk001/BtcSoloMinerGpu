"""Logging utilities for verbose logging support"""
import logging


def _vlog(logger, verbose, message):
    """Verbose logging helper - logs if verbose is True OR logger is at DEBUG level"""
    # Log if verbose flag is set OR if logger is at DEBUG level
    if verbose or logger.isEnabledFor(logging.DEBUG):
        logger.debug(message)


def get_verbose_logging(config: dict) -> bool:
    """Get verbose logging flag from config"""
    return config.get("logging", {}).get("verbose", False)

