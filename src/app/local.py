import asyncio
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from app.core.config import LocalConfig
from app.core.logger import setup_logging
from app.domain.pipeline import PhotoClusteringPipeline

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting local pipeline run...")
    config = LocalConfig("test")
    logger.info(f"Configuration loaded for job_id: {config.job_id}")

    # Use context managers for executors to ensure they are properly shut down
    with ThreadPoolExecutor() as thread_executor, ProcessPoolExecutor() as process_executor:
        logger.info("Executors initialized.")
        pipeline = PhotoClusteringPipeline(config, thread_executor, process_executor)
        await pipeline.run()
    
    logger.info("Local pipeline run finished.")


if __name__ == "__main__":
    asyncio.run(main())
