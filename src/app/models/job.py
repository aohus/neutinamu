from datetime import datetime

from sqlalchemy import JSON, Column, DateTime
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey, Integer, String, Text, desc
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.database import Base
from app.models.utils import generate_short_id
from app.schemas.enum import ExportStatus, JobStatus


class ClusterJob(Base):
    __tablename__ = "cluster_jobs"

    id = Column(String, primary_key=True, index=True, default=lambda: generate_short_id("clsJob"))
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"))
    error_message = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    finished_at = Column(DateTime, nullable=True)

    job = relationship("Job", back_populates="cluster_job")

    def __repr__(self) -> str:
        return f"<ClusterJob(id={self.id}, title={self.job.title})>"


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id = Column(String, primary_key=True, index=True, default=lambda: generate_short_id("expJob"))
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"))
    status = Column(SqlEnum(ExportStatus), default=ExportStatus.PENDING)
    cover_title = Column(String(255), nullable=True)
    cover_company_name = Column(String(255), nullable=True)
    labels = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.now)
    finished_at = Column(DateTime, nullable=True)
    pdf_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)

    job = relationship("Job", back_populates="export_job")

    def __repr__(self) -> str:
        return f"<ExportJob(id={self.id}, title={self.job.title}), status={self.status}>"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: generate_short_id("job"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    title = Column(String, nullable=False)
    status = Column(SqlEnum(JobStatus), default=JobStatus.CREATED)
    construction_type = Column(String(255), nullable=True)
    company_name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="jobs")
    clusters = relationship(
        "Cluster",
        back_populates="job",
        order_by="Cluster.order_index",
        cascade="all, delete-orphan",
    )
    photos = relationship("Photo", back_populates="job", cascade="all, delete-orphan")
    cluster_job = relationship("ClusterJob", back_populates="job", cascade="all, delete-orphan")
    export_job = relationship(
        "ExportJob",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by=desc(ExportJob.created_at),
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, title={self.title})>"
