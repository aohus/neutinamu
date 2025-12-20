from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repository.user import UserRepository


@pytest.fixture
def mock_db_session():
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.mark.asyncio
async def test_get_by_username(mock_db_session):
    repo = UserRepository(mock_db_session)

    # Mock result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = User(username="test")
    mock_db_session.execute.return_value = mock_result

    user = await repo.get_by_username("test")
    assert user.username == "test"
    mock_db_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_create_user(mock_db_session):
    repo = UserRepository(mock_db_session)
    user = User(username="new")

    result = await repo.create(user)

    assert result == user
    mock_db_session.add.assert_called_once_with(user)
