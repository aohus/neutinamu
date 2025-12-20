import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import configs

logger = logging.getLogger(__name__)

# Create async engine
# Set echo to True only if log level is DEBUG for verbose SQL logging
is_debug = configs.LOG_LEVEL.upper() == "DEBUG"
engine = create_async_engine(configs.DATABASE_URL, echo=False, future=True)


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
