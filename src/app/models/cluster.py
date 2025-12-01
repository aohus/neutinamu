from datetime import datetime

from app.db.database import Base
from app.models.utils import generate_short_id
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Cluster(Base):
    __tablename__ = "clusters"
    
    id = Column(String, primary_key=True, default=lambda: generate_short_id("cls"))
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    order_index = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    job = relationship("Job", back_populates="clusters")
    photos = relationship("Photo", back_populates="cluster")

    def __repr__(self) -> str:
        return f"<Cluster(id={self.id}, name={self.name})>"