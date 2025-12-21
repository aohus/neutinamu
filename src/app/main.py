import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# 내부 모듈 임포트
from app.api.api import api_router
from app.core.config import configs
from app.core.logger import setup_logging
from app.core.uvicorn_config import uvicorn_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


setup_logging()
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

    @app.get("/", tags=["Root"])
    async def read_root():
        return {"message": "Welcome!", "docs_url": "/docs"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}


def init_monitoring(app: FastAPI) -> None:
    # Prometheus 메트릭 노출
    Instrumentator().instrument(app).expose(app)


def init_log_filter() -> None:
    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "GET /metrics" not in record.getMessage()
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application lifespan...")
    app.state.thread_executor = ThreadPoolExecutor()
    app.state.process_executor = ProcessPoolExecutor()
    yield
    logger.info("Shutting down application lifespan...")
    app.state.thread_executor.shutdown(wait=False)
    app.state.process_executor.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Photo Clustering API",
        version="1.0.0",
        lifespan=lifespan,
    )
    init_routers(app)
    init_monitoring(app)
    init_log_filter()
    init_cors(app)
    return app


app = create_app()


if __name__ == "__main__":
    is_dev = configs.ENVIRONMENT != "production"
    
    uvicorn.run(
        app="app.main:app",
        host=configs.APP_HOST,
        port=configs.APP_PORT,
        reload=is_dev,
        **uvicorn_settings
    )
