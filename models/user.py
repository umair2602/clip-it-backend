"""
User models for the authentication system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, BeforeValidator, EmailStr, Field, field_validator


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
    DOWNLOADED = "downloaded"  # After download, before processing
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    PROCESSING_STARTED = "processing_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipModel(BaseModel):
    """Clip model within video"""
    id: str = Field(default_factory=lambda: str(ObjectId()))
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
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        json_encoders = {ObjectId: str}


class VideoModel(BaseModel):
    """Video model within user"""
    id: str = Field(default_factory=lambda: str(ObjectId()))
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
    clip_thumbnail_url: Optional[str] = None
    status: VideoStatus = VideoStatus.UPLOADING
    error_message: Optional[str] = None
    filename: str
    clips: List[ClipModel] = []
    process_task_id: Optional[str] = None  # Task ID for processing job (for task switching)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    processed_at: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}


class UserBase(BaseModel):
    """Base user model"""
    username: str = Field(..., min_length=3, max_length=50)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: EmailStr

    class Config:
        json_encoders = {ObjectId: str}


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=6, max_length=100)
    privacy_accepted: bool = Field(..., description="User must accept privacy policy")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate password"""
        if not v:
            raise ValueError("Password is required")
        if len(v.strip()) < 6:
            raise ValueError("Password must be at least 6 characters long")
        return v.strip()
    
    @field_validator('username', 'first_name', 'last_name')
    @classmethod
    def validate_string_fields(cls, v):
        """Validate and sanitize string fields"""
        if not v:
            raise ValueError("Field cannot be empty")
        
        clean_value = v.strip()
        if not clean_value:
            raise ValueError("Field cannot be empty after trimming whitespace")
        
        return clean_value
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Validate and normalize email"""
        if not v:
            raise ValueError("Email is required")
        
        clean_email = v.strip().lower()
        if not clean_email:
            raise ValueError("Email cannot be empty after trimming whitespace")
        
        return clean_email


class UserUpdate(BaseModel):
    """User update model"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[EmailStr] = None


class UserInDB(UserBase):
    """User model as stored in database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    hashed_password: str
    is_active: bool = True
    privacy_accepted: bool = True
    videos: List[VideoModel] = []
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class User(UserBase):
    """User model for API responses"""
    id: str
    is_active: bool
    privacy_accepted: bool
    videos: List[VideoModel] = []
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data model"""
    user_id: Optional[str] = None
    username: Optional[str] = None


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str
