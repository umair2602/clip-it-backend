"""
Standalone Clip model for clips collection
"""
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from bson import ObjectId
from pydantic import BaseModel, Field


def utc_now():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


class Clip(BaseModel):
    """Standalone Clip model for clips collection"""
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    video_id: str  # Foreign key to videos collection
    user_id: str   # Denormalized for quick user queries
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
        populate_by_name = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
