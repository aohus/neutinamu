import logging
from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

# Create async engine
# Set echo to True only if log level is DEBUG for verbose SQL logging
is_debug = settings.LOG_LEVEL.upper() == "DEBUG"
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=is_debug,
    future=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create base class for models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency for getting async database sessions."""
    logger.debug("Creating new database session.")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            logger.debug("Closing database session.")
            await session.close()