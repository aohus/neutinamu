import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# 1. 경로 설정 (최상단)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.api import api_router
from app.core.config import configs
from app.core.logger import setup_logging
from app.core.uvicorn_config import uvicorn_settings

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting application lifespan...")
    # ThreadPool은 입출력 작업에, ProcessPool은 CPU 연산에 사용
    app.state.thread_executor = ThreadPoolExecutor(max_workers=10)
    app.state.process_executor = ProcessPoolExecutor(max_workers=2) # Docker 자원에 맞춰 조절
    yield
    logger.info("🛑 Shutting down application lifespan...")
    app.state.thread_executor.shutdown(wait=False)
    app.state.process_executor.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Photo Clustering API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 미들웨어 및 라우터 초기화
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configs.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/", tags=["Root"])
    async def read_root():
        return {"message": "Welcome!", "docs_url": "/docs"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # 모니터링 설정 (/metrics 엔드포인트 자동 생성)
    Instrumentator().instrument(app).expose(app)

    # 로그 필터 설정
    init_log_filter()

    return app

def init_log_filter() -> None:
    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "/metrics" not in record.getMessage()
    
    # uvicorn 로그에서 metrics 호출 기록 제거
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


app = create_app()

if __name__ == "__main__":
    import uvicorn

    is_dev = configs.ENVIRONMENT != "production"
    
    run_config = {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 8000,
        **uvicorn_settings
    }
    
    if is_dev:
        run_config["reload"] = True
        run_config["workers"] = 1  # Reload 모드 강제 설정
        
    uvicorn.run(**run_config)
