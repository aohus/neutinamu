import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from app.main import app
from app.models.user import User

# Fix for "Task got bad yield" error with pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def mock_user():
    return User(
        user_id="123e4567-e89b-12d3-a456-426614174000",
        email="test@example.com",
        company_name="Test Corp"
    )
