import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Generator

from fastapi import Request

logger = logging.getLogger(__name__)


def get_thread_executor(request: Request) -> Generator[ThreadPoolExecutor, None, None]:
    logger.debug("Yielding thread executor from app state.")
    yield request.app.state.thread_executor


def get_process_executor(
    request: Request,
) -> Generator[ProcessPoolExecutor, None, None]:
    logger.debug("Yielding process executor from app state.")
    yield request.app.state.process_executor
