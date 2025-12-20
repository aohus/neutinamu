import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status

from app.api.deps import get_uow
from app.main import app
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse


@pytest.mark.asyncio
async def test_register(client):
    payload = {
        "user_id": "test_user_id",
        "email": "test@example.com",
        "password": "password123",
        "username": "testuser",
        "company_name": "Test Company",
    }

    # Mock response from service
    # UserResponse does not include email, is_active, is_superuser
    expected_response = UserResponse(
        user_id=uuid.uuid4(), username="testuser", company_name="Test Company", created_at="2024-01-01T00:00:00"
    )

    with patch("app.api.endpoints.auth.AuthService") as MockService:
        mock_service = MockService.return_value
        mock_service.register = AsyncMock(return_value=expected_response)

        response = await client.post("/api/auth/register", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["username"] == "testuser"
        # assert data["email"] == "test@example.com" # email is not in UserResponse


@pytest.mark.asyncio
async def test_login(client):
    payload = {"username": "test@example.com", "password": "password123"}
    expected_token = "test_access_token"

    with patch("app.api.endpoints.auth.AuthService") as MockService:
        mock_service = MockService.return_value
        mock_service.login = AsyncMock(return_value=expected_token)

        # Login expects form data (OAuth2PasswordRequestForm)
        response = await client.post(
            "/api/auth/login", data={"username": payload["username"], "password": payload["password"]}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"] == expected_token
        assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_get_me(client, mock_user):
    from app.api.endpoints.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        response = await client.get("/api/auth/me")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user_id"] == str(mock_user.user_id)
        # assert data["email"] == mock_user.email # UserResponse does not include email
    finally:
        del app.dependency_overrides[get_current_user]
