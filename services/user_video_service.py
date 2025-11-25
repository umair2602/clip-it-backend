"""
Service for managing user videos - migrated to use separate collections
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from database.connection import get_database

# Import from new class-based services
from services.video_service import video_service
from services.clip_service import create_clip, get_video_clips, get_user_clips

logger = logging.getLogger(__name__)


def utc_now():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


async def create_user_video(user_id: str, video_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a new video in videos collection.
    
    Args:
        user_id: User ID
        video_data: Video data dictionary
        
    Returns:
        str: Video ID if successful, None otherwise
    """
    try:
        logger.info(f"🔍 Creating video for user {user_id}")
        logger.info(f"   process_task_id: {video_data.get('process_task_id', 'NOT PROVIDED')}")
        
        # Create video using video_service
        video_id = await video_service.create_video(user_id, video_data)
        
        if video_id:
            logger.info(f"✅ Created video {video_id} for user {user_id}")
            return video_id
        else:
            logger.error(f"Failed to create video for user {user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating video for user {user_id}: {str(e)}")
        return None


async def update_user_video(user_id: str, video_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Update a video.
    
    Args:
        user_id: User ID
        video_id: Video ID
        update_data: Update data dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Remove None values
        clean_data = {k: v for k, v in update_data.items() if v is not None}
        result = await video_service.update_video(video_id, user_id, clean_data)
        return result is not None
    except Exception as e:
        logger.error(f"Error updating video {video_id}: {str(e)}")
        return False


async def update_video_s3_url(user_id: str, video_id: str, s3_url: str, s3_key: str) -> bool:
    """
    Update video S3 URL and key.
    
    Args:
        user_id: User ID
        video_id: Video ID
        s3_url: S3 URL
        s3_key: S3 key
        
    Returns:
        bool: True if successful, False otherwise
    """
    from models.video import VideoStatus
    return await update_user_video(user_id, video_id, {
        "s3_url": s3_url,
        "s3_key": s3_key,
        "status": VideoStatus.COMPLETED
    })


async def get_user_video(user_id: str, video_id: str):
    """
    Get a specific video for a user.
    
    Args:
        user_id: User ID
        video_id: Video ID
        
    Returns:
        Video object if found, None otherwise
    """
    try:
        return await video_service.get_video_by_id(video_id, user_id)
    except Exception as e:
        logger.error(f"Error getting video {video_id} for user {user_id}: {str(e)}")
        return None


async def get_user_video_by_video_id(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a video by its ID.
    
    Args:
        video_id: Video ID
        
    Returns:
        Dict: Video data if found, None otherwise
    """
    try:
        video = await video_service.get_video(video_id)
        if video:
            return video.dict() if hasattr(video, 'dict') else video
        return None
    except Exception as e:
        logger.error(f"Error getting video {video_id}: {str(e)}")
        return None


async def get_user_videos(user_id: str, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """
    Get all videos for a user with pagination.
    
    Args:
        user_id: User ID
        page: Page number
        page_size: Page size
        
    Returns:
        Dict: Videos and pagination info
    """
    try:
        return await video_service.get_user_videos(user_id, page=page, page_size=page_size)
    except Exception as e:
        logger.error(f"Error getting videos for user {user_id}: {str(e)}")
        return {"videos": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}


async def add_clip_to_video(user_id: str, video_id: str, clip_data: Dict[str, Any]) -> Optional[str]:
    """
    Add a clip to a video in clips collection.
    
    Args:
        user_id: User ID
        video_id: Video ID
        clip_data: Clip data dictionary
        
    Returns:
        str: Clip ID if successful, None otherwise
    """
    try:
        logger.info(f"Adding clip to video {video_id} for user {user_id}")
        logger.info(f"Clip data: {clip_data}")
        
        # Create clip in clips collection
        clip_id = await create_clip(video_id, user_id, clip_data)
        
        if clip_id:
            logger.info(f"✅ Created clip {clip_id} for video {video_id}")
            return clip_id
        else:
            logger.error(f"Failed to create clip for video {video_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error adding clip to video {video_id}: {str(e)}")
        return None


async def get_user_clips(user_id: str, video_id: Optional[str] = None) -> List:
    """
    Get clips for a user, optionally filtered by video.
    
    Args:
        user_id: User ID
        video_id: Optional video ID to filter by
        
    Returns:
        List of clips
    """
    try:
        if video_id:
            return await get_video_clips(video_id)
        else:
            result = await video_service.get_user_clips(user_id)
            return result.get('clips', [])
    except Exception as e:
        logger.error(f"Error getting clips for user {user_id}: {str(e)}")
        return []


async def delete_user_video(user_id: str, video_id: str) -> bool:
    """
    Delete a video and its clips.
    
    Args:
        user_id: User ID
        video_id: Video ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        return await video_service.delete_video(video_id, user_id)
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        return False