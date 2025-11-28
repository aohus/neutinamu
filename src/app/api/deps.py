from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Generator

from fastapi import Request


def get_thread_executor(request: Request) -> Generator[ThreadPoolExecutor, None, None]:
    yield request.app.state.thread_executor


def get_process_executor(
    request: Request,
) -> Generator[ProcessPoolExecutor, None, None]:
    yield request.app.state.process_executor
