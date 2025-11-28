import logging

from app.core.security import decode_access_token
from app.db.database import get_db
from app.models.user import User
from app.schemas.user import Token, UserCreate, UserResponse
from app.services.auth import AuthService
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
logger = logging.getLogger(__name__)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    if payload is None:
        logger.warning("Could not validate credentials: Token decoding failed.")
        raise credentials_exception
    
    user_id: str = payload.get("sub")
    if user_id is None:
        logger.warning("Could not validate credentials: User ID not in token payload.")
        raise credentials_exception
    
    logger.debug(f"Fetching user {user_id} from database.")
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        logger.warning(f"User {user_id} not found in database.")
        raise credentials_exception
    
    logger.debug(f"Authenticated user: {user.username}")
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    service = AuthService(db)
    new_user = await service.register(user_data)
    return new_user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    service = AuthService(db)
    access_token = await service.login(form_data)
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    logger.info(f"Fetching profile for user: {current_user.username}")
    return current_user
