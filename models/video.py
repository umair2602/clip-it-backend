"""
Video models for tracking user uploads, YouTube downloads, and processing results.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field, BeforeValidator
from bson import ObjectId
from enum import Enum


def utc_now():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


def validate_object_id(v):
    """Validate ObjectId"""
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return v
        raise ValueError("Invalid ObjectId")
    raise ValueError("Invalid ObjectId")


PyObjectId = Annotated[str, BeforeValidator(validate_object_id)]


class VideoType(str, Enum):
    """Video source types"""
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class VideoStatus(str, Enum):
    """Video processing status"""
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoBase(BaseModel):
    """Base video model"""
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

    class Config:
        json_encoders = {ObjectId: str}


class VideoCreate(VideoBase):
    """Video creation model"""
    user_id: str
    filename: str


class VideoUpdate(BaseModel):
    """Video update model"""
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: Optional[VideoStatus] = None
    error_message: Optional[str] = None


class VideoInDB(VideoBase):
    """Video model as stored in database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: str
    filename: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    processed_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Video(VideoBase):
    """Video model for API responses"""
    id: str
    user_id: str
    filename: str
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None


class ClipBase(BaseModel):
    """Base clip model"""
    video_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: float
    end_time: float
    duration: float
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    transcription: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {ObjectId: str}


class ClipCreate(ClipBase):
    """Clip creation model"""
    user_id: str


class ClipInDB(ClipBase):
    """Clip model as stored in database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Clip(ClipBase):
    """Clip model for API responses"""
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class VideoHistoryResponse(BaseModel):
    """Response model for user video history"""
    videos: List[Video]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class ClipHistoryResponse(BaseModel):
    """Response model for user clip history"""
    clips: List[Clip]
    total_count: int
    page: int
    page_size: int
    total_pages: int 