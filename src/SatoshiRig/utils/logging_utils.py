"""Logging utilities for verbose logging support"""
import logging


def _vlog(logger, verbose, message):
    """Verbose logging helper - logs if verbose is True"""
    if verbose:
        logger.debug(message)


def get_verbose_logging(config: dict) -> bool:
    """Get verbose logging flag from config"""
    return config.get("logging", {}).get("verbose", False)

