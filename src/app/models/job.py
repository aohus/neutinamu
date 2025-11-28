import uuid
from datetime import datetime

from app.db.database import Base
from app.models.utils import generate_short_id
from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: generate_short_id("job"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, default="CREATED")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="jobs")
    clusters = relationship("Cluster", back_populates="job", cascade="all, delete-orphan")
    photos = relationship("Photo", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, title={self.title})>"