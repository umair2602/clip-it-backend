"""
Service layer for Clip operations
"""
import logging
from typing import Optional, List, Dict, Any
from bson import ObjectId
from database.connection import get_database
from models.clip import Clip

logger = logging.getLogger(__name__)


async def create_clip(video_id: str, user_id: str, clip_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a new clip in clips collection
    
    Args:
        video_id: Video ID this clip belongs to
        user_id: User ID who owns the clip
        clip_data: Clip data dictionary
        
    Returns:
        str: Clip ID if successful, None otherwise
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        # Create clip model
        clip = Clip(
            video_id=video_id,
            user_id=user_id,
            title=clip_data.get('title'),
            description=clip_data.get('description'),
            start_time=clip_data['start_time'],
            end_time=clip_data['end_time'],
            duration=clip_data['end_time'] - clip_data['start_time'],
            s3_key=clip_data.get('s3_key'),
            s3_url=clip_data.get('s3_url'),
            thumbnail_url=clip_data.get('thumbnail_url'),
            transcription=clip_data.get('transcription'),
            summary=clip_data.get('summary'),
            tags=clip_data.get('tags'),
            metadata=clip_data.get('metadata')
        )
        
        # Insert into collection
        clip_dict = clip.dict(by_alias=True)
        result = clips_collection.insert_one(clip_dict)
        
        logger.info(f"✅ Created clip {clip.id} for video {video_id}")
        return clip.id
        
    except Exception as e:
        logger.error(f"❌ Error creating clip for video {video_id}: {str(e)}")
        return None


async def get_clip(clip_id: str) -> Optional[Clip]:
    """
    Get a clip by ID
    
    Args:
        clip_id: Clip ID
        
    Returns:
        Clip: Clip object if found, None otherwise
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        clip_data = clips_collection.find_one({"_id": clip_id})
        
        if clip_data:
            return Clip(**clip_data)
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting clip {clip_id}: {str(e)}")
        return None


async def get_video_clips(video_id: str) -> List[Clip]:
    """
    Get all clips for a video
    
    Args:
        video_id: Video ID
        
    Returns:
        List[Clip]: List of clips
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        cursor = clips_collection.find({"video_id": video_id}).sort("start_time", 1)
        
        clips = [Clip(**clip_data) for clip_data in cursor]
        return clips
        
    except Exception as e:
        logger.error(f"❌ Error getting clips for video {video_id}: {str(e)}")
        return []


async def get_user_clips(user_id: str, skip: int = 0, limit: int = 20) -> List[Clip]:
    """
    Get all clips for a user with pagination
    
    Args:
        user_id: User ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List[Clip]: List of clips
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        cursor = clips_collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        
        clips = [Clip(**clip_data) for clip_data in cursor]
        return clips
        
    except Exception as e:
        logger.error(f"❌ Error getting clips for user {user_id}: {str(e)}")
        return []


async def update_clip(clip_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update a clip
    
    Args:
        clip_id: Clip ID
        updates: Dictionary of fields to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        from models.clip import utc_now
        updates['updated_at'] = utc_now()
        
        result = clips_collection.update_one(
            {"_id": clip_id},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"✅ Updated clip {clip_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ Error updating clip {clip_id}: {str(e)}")
        return False


async def delete_clip(clip_id: str) -> bool:
    """
    Delete a clip
    
    Args:
        clip_id: Clip ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_database()
        clips_collection = db.clips
        
        result = clips_collection.delete_one({"_id": clip_id})
        
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted clip {clip_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ Error deleting clip {clip_id}: {str(e)}")
        return False
