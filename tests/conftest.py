import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from app.main import app
from app.models.user import User
from app.api.deps import get_uow
from app.db.database import get_db

# Fix for "Task got bad yield" error with pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_db_session():
    return AsyncMock()

@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.users = MagicMock()
    uow.jobs = MagicMock()
    uow.clusters = MagicMock()
    uow.photos = MagicMock()
    uow.commit = AsyncMock()
    uow.flush = AsyncMock()
    uow.refresh = AsyncMock()
    
    # Setup some default behaviors for repositories to avoid NoneType errors
    uow.users.get_by_username = AsyncMock(return_value=None)
    uow.jobs.get_by_id = AsyncMock(return_value=None)
    
    return uow

@pytest_asyncio.fixture
async def client(mock_db_session, mock_uow) -> AsyncGenerator[AsyncClient, None]:
    # Override dependencies
    async def override_get_db():
        yield mock_db_session

    async def override_get_uow():
        yield mock_uow

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_uow] = override_get_uow
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides = {}

import uuid

# ... imports ...

@pytest.fixture
def mock_user():
    from datetime import datetime
    return User(
        user_id=uuid.uuid4(),
        company_name="Test Corp",
        username="testuser",
        password_hash="hashed",
        created_at=datetime.now()
    )