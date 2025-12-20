import pytest
from unittest.mock import AsyncMock, patch
from fastapi import status
from app.schemas.user import UserCreate, UserResponse

@pytest.mark.asyncio
async def test_register(client):
    payload = {
        "user_id": "test_user_id",
        "email": "test@example.com",
        "password": "password123",
        "username": "testuser",
        "company_name": "Test Company"
    }
    
    # Mock response from service
    expected_response = UserResponse(
        user_id="test_user_id",
        email="test@example.com",
        username="testuser",
        company_name="Test Company",
        is_active=True,
        is_superuser=False
    )

    with patch("app.api.endpoints.auth.AuthService") as MockService:
        mock_service = MockService.return_value
        mock_service.register = AsyncMock(return_value=expected_response)
        
        response = await client.post("/api/auth/register", json=payload)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"

@pytest.mark.asyncio
async def test_login(client):
    payload = {
        "username": "test@example.com",
        "password": "password123"
    }
    expected_token = "test_access_token"

    with patch("app.api.endpoints.auth.AuthService") as MockService:
        mock_service = MockService.return_value
        mock_service.login = AsyncMock(return_value=expected_token)
        
        # Login expects form data (OAuth2PasswordRequestForm)
        response = await client.post(
            "/api/auth/login", 
            data={"username": payload["username"], "password": payload["password"]}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"] == expected_token
        assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_get_me(client, mock_user):
    with patch("app.api.endpoints.auth.get_current_user", return_value=mock_user):
        response = await client.get("/api/auth/me")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # mock_user has user_id="...", email="...", company_name="..."
        assert data["user_id"] == mock_user.user_id
        assert data["email"] == mock_user.email
