"""Pipeline logging utilities."""
import logging
import sys
from pathlib import Path


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
    ch = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8",
             closefd=False, buffering=1)
    )
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
