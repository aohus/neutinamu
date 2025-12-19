import logging
import os  # Added os import
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.api import api_router
from app.core.config import settings
from app.core.logger import setup_logging

from prometheus_fastapi_instrumentator import Instrumentator


# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "assets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    logger.info("Starting application lifespan...")
    # Startup: Initialize executors and store them in the app state
    app.state.thread_executor = ThreadPoolExecutor()
    app.state.process_executor = ProcessPoolExecutor()
    logger.info("Executors initialized and data directory ensured.")
    yield

    # Shutdown: Stop the executors
    logger.info("Shutting down application lifespan...")
    app.state.thread_executor.shutdown(wait=False)
    app.state.process_executor.shutdown(wait=False)
    logger.info("Executors shut down.")


app = FastAPI(
    title="Photo Clustering API",
    description="An API to upload photos and cluster them based on various criteria.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

app.include_router(api_router, prefix="/api")

# Conditionally mount StaticFiles based on STORAGE_TYPE
if settings.STORAGE_TYPE == "local":
    # Ensure MEDIA_ROOT exists for local storage
    os.makedirs(MEDIA_ROOT, exist_ok=True)
    app.mount(
        "/api/uploads",              
        StaticFiles(directory=MEDIA_ROOT),
        name="uploads",
    )
else:
    logger.info(f"STORAGE_TYPE is {settings.STORAGE_TYPE}, StaticFiles for /api/uploads will not be mounted.")


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to the Photo Clustering API!",
        "docs_url": "/docs",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)