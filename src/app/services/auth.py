import logging
from datetime import timedelta

from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.common.uow import UnitOfWork
from app.core.config import settings
from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from app.models.user import User
from app.schemas.user import UserCreate

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def register(self, user_data: UserCreate):
        """Register a new user."""
        logger.info(f"Registration attempt for username: {user_data.username}")
        # Check if username exists
        if await self.uow.users.get_by_username(user_data.username):
            logger.warning(f"Registration failed: Username '{user_data.username}' already registered.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            company_name=user_data.company_name,
            password_hash=hashed_password
        )
        
        await self.uow.users.create(new_user)
        await self.uow.commit()
        await self.uow.refresh(new_user)
        
        logger.info(f"User '{new_user.username}' registered successfully with ID: {new_user.user_id}")
        return new_user

    async def login(self, form_data: OAuth2PasswordRequestForm):
        """Login and get access token."""
        logger.info(f"Login attempt for username: {form_data.username}")
        # Find user by username
        user = await self.uow.users.get_by_username(form_data.username)
        
        if not user or not verify_password(form_data.password, user.password_hash):
            logger.warning(f"Login failed for username: {form_data.username}. Incorrect username or password.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.user_id)},
            expires_delta=access_token_expires
        )
        logger.info(f"Login successful for user: {user.username}")
        return access_token

