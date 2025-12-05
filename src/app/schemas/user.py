from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=6, max_length=50)
    password: str = Field(..., min_length=6)
    company_name: Optional[str] = Field(..., min_length=2, max_length=50)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: UUID
    username: str
    company_name: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None


class UserPreferenceCreate(BaseModel):
    search_conditions: Dict[str, Any]


class UserPreferenceResponse(BaseModel):
    preference_id: UUID
    user_id: UUID
    search_conditions: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SavedSearchCreate(BaseModel):
    search_name: str = Field(..., min_length=1, max_length=100)
    filters: Dict[str, Any]


class SavedSearchResponse(BaseModel):
    search_id: UUID
    user_id: UUID
    search_name: str
    filters: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True


class BookmarkCreate(BaseModel):
    bid_notice_no: str
    bid_notice_name: str
    notes: Optional[str] = None


class BookmarkResponse(BaseModel):
    bookmark_id: UUID
    user_id: UUID
    bid_notice_no: str
    bid_notice_name: str
    notes: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True