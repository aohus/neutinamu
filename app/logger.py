import logging
import sys

def setup_logging(level=logging.INFO):
    """Set up the root logger."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_logger(name: str):
    """Get a logger instance."""
    return logging.getLogger(name)
