# app/core/logging.py
import logging
import sys
from .config import settings

def configure_logging() -> None:
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=settings.LOG_LEVEL, handlers=[handler])
