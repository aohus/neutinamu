import asyncio
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from app.config import AppConfig
from app.logger import get_logger, setup_logging
from app.pipeline import PhotoClusteringPipeline

logger = get_logger(__name__)

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def main():
    setup_logging()
    config = AppConfig()
    
    # Use context managers for executors to ensure they are properly shut down
    with ThreadPoolExecutor() as thread_executor, ProcessPoolExecutor() as process_executor:
        pipeline = PhotoClusteringPipeline(config, thread_executor, process_executor)
        await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
