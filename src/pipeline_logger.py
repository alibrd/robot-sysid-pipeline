"""Pipeline logging utilities."""
import logging
import sys
from pathlib import Path

# Re-used across setup_logger calls: re-wrapping sys.stdout's fd on every
# call leaks stream objects when several runners set up loggers in one
# process (pipeline + validation stages).
_CONSOLE_STREAM = None


def _console_stream():
    global _CONSOLE_STREAM
    if _CONSOLE_STREAM is None or _CONSOLE_STREAM.closed:
        # UTF-8 wrapper to avoid UnicodeEncodeError on Windows consoles
        _CONSOLE_STREAM = open(
            sys.stdout.fileno(), mode="w", encoding="utf-8",
            closefd=False, buffering=1,
        )
    return _CONSOLE_STREAM


def setup_logger(output_dir: str,
                 name: str = "sysid_pipeline",
                 log_filename: str = "pipeline.log") -> logging.Logger:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / log_filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # File handler: UTF-8 so special characters are preserved in the log
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler: force UTF-8 to avoid UnicodeEncodeError on Windows
    ch = logging.StreamHandler(_console_stream())
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
