from enum import Enum


class JobStatus(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DELETED = "DELETED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class ExportStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    EXPORTED = "EXPORTED"