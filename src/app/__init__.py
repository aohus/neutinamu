import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.api import api_router
from app.core.config import configs
from app.core.logger import setup_logging

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# setup_logging()  <-- Moved to main.py
logger = logging.getLogger(__name__)


def init_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configs.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def init_routers(app: FastAPI) -> None:
    app.include_router(api_router, prefix="/api")
    
    # Add Health and Root endpoints directly
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


def init_monitoring(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(app)
    # app.add_middleware(PrometheusHTTPMiddleware, app_name=config.APP_NAME)
    # app.add_middleware(PrometheusWebSocketMiddleware, app_name=config.APP_NAME)
    # app.add_middleware(ProfilerMiddleware, interval=0.01)
    # setup_monitoring()
    # Setting OpenTelemetry exporter
    # setting_otlp(app, config.APP_NAME, config.OTEL_EXPORTER_OTLP_ENDPOINT)

def init_log_filter() -> None:
    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "GET /metrics" not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Photo Clustering API",
        description="An API to upload photos and cluster them based on various criteria.",
        version="1.0.0",
        lifespan=lifespan,
    )

    init_routers(app=app)
    init_monitoring(app=app)
    init_log_filter()
    init_cors(app=app)
    return app
