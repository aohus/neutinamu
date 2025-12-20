import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from app.services.auth import AuthService
from app.schemas.user import UserCreate
from app.models.user import User

@pytest.fixture
def mock_uow():
    uow = MagicMock()
    uow.users = MagicMock()
    uow.commit = AsyncMock()
    uow.refresh = AsyncMock()
    return uow

@pytest.mark.asyncio
async def test_register_success(mock_uow):
    service = AuthService(mock_uow)
    user_data = UserCreate(
        username="newuser",
        email="new@example.com",
        password="password",
        company_name="Company"
    )
    
    mock_uow.users.get_by_username = AsyncMock(return_value=None)
    mock_uow.users.create = AsyncMock()
    
    user = await service.register(user_data)
    
    assert user.username == "newuser"
    mock_uow.users.create.assert_called_once()
    mock_uow.commit.assert_called_once()

@pytest.mark.asyncio
async def test_register_existing_user(mock_uow):
    service = AuthService(mock_uow)
    user_data = UserCreate(username="existing", email="e@e.com", password="password123", company_name="Company")
    
    mock_uow.users.get_by_username = AsyncMock(return_value=User(username="existing"))
    
    with pytest.raises(HTTPException) as exc:
        await service.register(user_data)
    assert exc.value.status_code == 400

@pytest.mark.asyncio
async def test_login_success(mock_uow):
    from app.core.security import get_password_hash
    service = AuthService(mock_uow)
    
    password = "password"
    hashed = get_password_hash(password)
    user = User(user_id="u1", username="user", password_hash=hashed)
    
    mock_uow.users.get_by_username = AsyncMock(return_value=user)
    
    form_data = MagicMock()
    form_data.username = "user"
    form_data.password = password
    
    token = await service.login(form_data)
    assert token is not None

@pytest.mark.asyncio
async def test_login_wrong_password(mock_uow):
    from app.core.security import get_password_hash
    service = AuthService(mock_uow)
    
    user = User(user_id="u1", username="user", password_hash=get_password_hash("password"))
    mock_uow.users.get_by_username = AsyncMock(return_value=user)
    
    form_data = MagicMock()
    form_data.username = "user"
    form_data.password = "wrong"
    
    with pytest.raises(HTTPException) as exc:
        await service.login(form_data)
    assert exc.value.status_code == 401
