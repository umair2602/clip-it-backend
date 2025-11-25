"""
Standalone Video model for videos collection
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from bson import ObjectId
from pydantic import BaseModel, Field

# Re-export Clip for compatibility with routes
from models.clip import Clip


def utc_now():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


class VideoType(str, Enum):
    """Video source types"""
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class VideoStatus(str, Enum):
    """Video processing status"""
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Video(BaseModel):
    """Standalone Video model for videos collection"""
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    user_id: str  # Foreign key to users collection
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[float] = None  # Duration in seconds
    file_size: Optional[int] = None  # File size in bytes
    content_type: Optional[str] = None
    video_type: VideoType = VideoType.UPLOAD
    source_url: Optional[str] = None  # For YouTube videos
    s3_key: Optional[str] = None  # S3 storage key
    s3_url: Optional[str] = None  # S3 public URL
    thumbnail_url: Optional[str] = None
    status: VideoStatus = VideoStatus.UPLOADING
    error_message: Optional[str] = None
    filename: str = ""
    process_task_id: Optional[str] = None  # Task ID for processing job
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    processed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}


class VideoCreate(BaseModel):
    """Model for creating a new video"""
    user_id: str
    filename: str
    title: Optional[str] = None
    description: Optional[str] = None
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: VideoStatus = VideoStatus.UPLOADING
    video_type: VideoType = VideoType.UPLOAD
    source_url: Optional[str] = None
    process_task_id: Optional[str] = None


class VideoUpdate(BaseModel):
    """Model for updating a video"""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[VideoStatus] = None
    s3_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error_message: Optional[str] = None


class VideoHistoryResponse(BaseModel):
    """Response model for video history"""
    user_id: str
    videos: list
    total_count: int
    page: int
    page_size: int
    total_pages: int


class ClipHistoryResponse(BaseModel):
    """Response model for clip history"""
    user_id: str
    clips: list
    total_count: int
    page: int
    page_size: int
    total_pages: int
