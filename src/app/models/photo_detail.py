from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

from app.db.database import Base
from app.models.utils import generate_short_id

class PhotoDetail(Base):
    __tablename__ = "photo_details"

    id = Column(String, primary_key=True, default=lambda: generate_short_id("phd"))
    photo_id = Column(String, ForeignKey("photos.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    device = Column(String, nullable=True)
    focal_length = Column(Float, nullable=True)
    exposure_time = Column(Float, nullable=True)
    iso = Column(Integer, nullable=True)
    flash = Column(Integer, nullable=True)
    orientation = Column(Integer, nullable=True)
    gps_img_direction = Column(Float, nullable=True)

    photo = relationship("Photo", back_populates="detail")

    def __repr__(self) -> str:
        return f"<PhotoDetail(id={self.id}, photo_id={self.photo_id}, device={self.device})>"
