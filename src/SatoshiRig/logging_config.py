import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(
    level: str = None, log_file: str = None, verbose: bool = False
) -> None:
    """Configure root logging for the application.

    - `level`: string like 'INFO' or 'DEBUG'. If None, will use env LOG_LEVEL or 'INFO'.
    - `log_file`: optional path for rotating file logging. If None, only stdout/stderr used.
    - `verbose`: if True, enables more verbose internal debug logs.
    """
    # Resolve defaults from environment
    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    log_file = log_file or os.environ.get("LOG_FILE")

    level_const = getattr(logging, level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level_const)

    # Remove existing file handlers if any
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    # Ensure a StreamHandler exists
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(level_const)
        sh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(sh)

    # Add a RotatingFileHandler if requested
    if log_file:
        try:
            fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setLevel(level_const)
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            root.addHandler(fh)
        except Exception as e:
            logging.getLogger("SatoshiRig.logging_config").warning(
                f"Failed to create file handler {log_file}: {e}"
            )

    # Optionally enable very verbose internal debug logs
    if verbose:
        logging.getLogger("SatoshiRig").setLevel(logging.DEBUG)
