from datetime import datetime

from app.db.database import Base
from app.models.utils import generate_short_id
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Photo(Base):
    __tablename__ = "photos"
    
    id = Column(String, primary_key=True, default=lambda: generate_short_id("pho"))
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    cluster_id = Column(String, ForeignKey("clusters.id"), nullable=True)
    
    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    thumbnail_path = Column(String, nullable=True)
    order_index = Column(Integer, nullable=True)
    
    # Metadata
    meta_lat = Column(Float, nullable=True)
    meta_lon = Column(Float, nullable=True)
    meta_timestamp = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    job = relationship("Job", back_populates="photos")
    cluster = relationship("Cluster", back_populates="photos")

    def __repr__(self) -> str:
        return f"<Photo(id={self.id}, filename={self.original_filename})>"