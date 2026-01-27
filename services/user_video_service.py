"""
Service for managing user videos with the new single table structure.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from database.connection import get_database
from models.user import UserInDB, VideoModel, ClipModel, VideoType, VideoStatus
from utils.s3_storage import s3_client

logger = logging.getLogger(__name__)


def utc_now():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


async def create_user_video(user_id: str, video_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a new video for a user.
    
    Args:
        user_id: User ID
        video_data: Video data dictionary
        
    Returns:
        str: Video ID if successful, None otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Create video model
        logger.info(f"🔍 Creating video model with data:")
        logger.info(f"   process_task_id in video_data: {video_data.get('process_task_id', 'NOT PROVIDED')}")
        logger.info(f"   job_id in video_data: {video_data.get('job_id', 'NOT PROVIDED')}")
        
        video = VideoModel(
            id=video_data.get('id', str(ObjectId())),  # Use provided ID or generate new one
            title=video_data.get('title'),
            description=video_data.get('description'),
            duration=video_data.get('duration'),
            file_size=video_data.get('file_size'),
            content_type=video_data.get('content_type'),
            video_type=video_data.get('video_type', VideoType.UPLOAD),
            source_url=video_data.get('source_url'),
            s3_key=video_data.get('s3_key'),
            s3_url=video_data.get('s3_url'),
            thumbnail_url=video_data.get('thumbnail_url'),
            status=video_data.get('status', VideoStatus.UPLOADING),
            error_message=video_data.get('error_message'),
            filename=video_data.get('filename', ''),
            clips=[],
            process_task_id=video_data.get('process_task_id'),  # Include process_task_id!
            created_at=utc_now(),
            updated_at=utc_now()
        )
        
        logger.info(f"📝 Video model created with process_task_id: {video.process_task_id}")
        
        # Add video to user's videos array, but only if it doesn't already exist (Idempotency ✅)
        result = users_collection.update_one(
            {
                "_id": ObjectId(user_id),
                "videos.id": {"$ne": video.id}  # Only push if id is not present
            },
            {
                "$push": {"videos": video.dict()},
                "$set": {"updated_at": utc_now()}
            }
        )

        
        if result.modified_count > 0:
            logger.info(f"✅ Created video {video.id} for user {user_id}")
            logger.info(f"   Process Task ID saved: {video.process_task_id}")
            
            # Verify it was actually saved
            verify_user = users_collection.find_one({"_id": ObjectId(user_id)})
            if verify_user:
                saved_video = None
                for v in verify_user.get('videos', []):
                    if v.get('id') == video.id:
                        saved_video = v
                        break
                if saved_video:
                    logger.info(f"🔍 VERIFICATION: Video in DB has process_task_id: {saved_video.get('process_task_id', 'MISSING')}")
                else:
                    logger.error(f"❌ VERIFICATION FAILED: Could not find video {video.id} in user's videos array")
            
            return video.id
        else:
            logger.error(f"Failed to create video for user {user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating video for user {user_id}: {str(e)}")
        return None


async def update_user_video(user_id: str, video_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Update a video for a user.
    
    Args:
        user_id: User ID
        video_id: Video ID
        update_data: Update data dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Prepare update fields
        update_fields = {}
        for key, value in update_data.items():
            if value is not None:
                update_fields[f"videos.$.{key}"] = value
        
        # Add updated_at timestamp
        update_fields["videos.$.updated_at"] = utc_now()
        update_fields["updated_at"] = utc_now()
        
        # Update the video
        result = users_collection.update_one(
            {
                "_id": ObjectId(user_id),
                "videos.id": video_id
            },
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated video {video_id} for user {user_id}")
            return True
        else:
            logger.warning(f"No video {video_id} found for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating video {video_id} for user {user_id}: {str(e)}")
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
    return await update_user_video(user_id, video_id, {
        "s3_url": s3_url,
        "s3_key": s3_key,
        "status": VideoStatus.COMPLETED
    })


async def get_user_video(user_id: str, video_id: str) -> Optional[VideoModel]:
    """
    Get a specific video for a user.
    
    Args:
        user_id: User ID
        video_id: Video ID
        
    Returns:
        VideoModel: Video if found, None otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        user = users_collection.find_one(
            {"_id": ObjectId(user_id)},
            {"videos": {"$elemMatch": {"id": video_id}}}
        )
        
        if user and user.get('videos'):
            video_data = user['videos'][0]
            return VideoModel(**video_data)
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting video {video_id} for user {user_id}: {str(e)}")
        return None


async def get_user_video_by_video_id(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a video by its ID across all users.
    
    Args:
        video_id: Video ID
        
    Returns:
        Dict: Video data with user_id if found, None otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Find any user that has this video
        user = users_collection.find_one({
            "videos.id": video_id
        })
        
        if user:
            # Find the specific video in the user's videos array
            for video in user.get('videos', []):
                if video.get('id') == video_id:
                    video_data = video.copy()
                    video_data['user_id'] = str(user['_id'])
                    return video_data
        
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
        db = get_database()
        users_collection = db.users
        
        # Get user with videos
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return {"videos": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}
        
        videos = user.get('videos', [])
        
        # Sort videos by created_at descending (newest first)
        videos = sorted(videos, key=lambda v: v.get('created_at', datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        
        total_count = len(videos)
        
        # Apply pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_videos = videos[start_index:end_index]
        
        # Convert to VideoModel objects
        video_models = [VideoModel(**video) for video in paginated_videos]
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "videos": video_models,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error getting videos for user {user_id}: {str(e)}")
        return {"videos": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}


async def add_clip_to_video(user_id: str, video_id: str, clip_data: Dict[str, Any]) -> Optional[str]:
    """
    Add a clip to a video.
    
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
        
        db = get_database()
        users_collection = db.users
        
        # Check if user exists
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.error(f"User {user_id} not found")
            return None
        
        # Check if video exists in user's videos
        video_found = False
        for video in user.get('videos', []):
            if video.get('id') == video_id:
                video_found = True
                break
        
        if not video_found:
            logger.error(f"Video {video_id} not found in user {user_id}'s videos")
            logger.info(f"Available videos: {[v.get('id') for v in user.get('videos', [])]}")
            return None
        
        # Create clip model
        clip = ClipModel(
            id=str(ObjectId()),
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
            metadata=clip_data.get('metadata'),
            created_at=utc_now(),
            updated_at=utc_now()
        )
        
        logger.info(f"Created clip model with ID: {clip.id}")
        logger.info(f"Clip dict: {clip.dict()}")
        
        # Add clip to video's clips array
        # First, check if a clip with the same start_time and end_time already exists
        existing_clip = users_collection.find_one({
            "_id": ObjectId(user_id),
            "videos": {
                "$elemMatch": {
                    "id": video_id,
                    "clips": {
                        "$elemMatch": {
                            "start_time": clip.start_time,
                            "end_time": clip.end_time
                        }
                    }
                }
            }
        })
        
        if existing_clip:
            logger.info(f"⏭️ Clip with start_time={clip.start_time} and end_time={clip.end_time} already exists, skipping")
            # Return existing clip ID (find it in the document)
            for video in existing_clip.get('videos', []):
                if video.get('id') == video_id:
                    for existing in video.get('clips', []):
                        if existing.get('start_time') == clip.start_time and existing.get('end_time') == clip.end_time:
                            return existing.get('id')
            return None
        
        # Now push the new clip
        result = users_collection.update_one(
            {
                "_id": ObjectId(user_id),
                "videos.id": video_id
            },
            {
                "$push": {"videos.$.clips": clip.dict()},
                "$set": {
                    "videos.$.updated_at": utc_now(),
                    "updated_at": utc_now()
                }
            }
        )

        
        logger.info(f"Update result: {result.modified_count} documents modified")
        logger.info(f"Update result matched: {result.matched_count} documents matched")
        
        if result.modified_count > 0:
            logger.info(f"✅ Added clip {clip.id} to video {video_id} for user {user_id}")
            return clip.id
        else:
            logger.error(f"❌ Failed to add clip to video {video_id} for user {user_id}")
            logger.error(f"Matched count: {result.matched_count}, Modified count: {result.modified_count}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error adding clip to video {video_id} for user {user_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


async def update_clip_in_video(user_id: str, video_id: str, clip_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Update a clip in a video.
    
    Args:
        user_id: User ID
        video_id: Video ID
        clip_id: Clip ID
        update_data: Update data dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Prepare update fields
        update_fields = {}
        for key, value in update_data.items():
            if value is not None:
                update_fields[f"videos.$.clips.$.{key}"] = value
        
        # Add updated_at timestamp
        update_fields["videos.$.clips.$.updated_at"] = utc_now()
        update_fields["videos.$.updated_at"] = utc_now()
        update_fields["updated_at"] = utc_now()
        
        # Update the clip
        result = users_collection.update_one(
            {
                "_id": ObjectId(user_id),
                "videos.id": video_id,
                "videos.clips.id": clip_id
            },
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated clip {clip_id} in video {video_id} for user {user_id}")
            return True
        else:
            logger.warning(f"No clip {clip_id} found in video {video_id} for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating clip {clip_id} in video {video_id} for user {user_id}: {str(e)}")
        return False


async def get_user_clips(user_id: str, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """
    Get all clips for a user with pagination.
    
    Args:
        user_id: User ID
        page: Page number
        page_size: Page size
        
    Returns:
        Dict: Clips and pagination info
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Get user with videos and clips
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return {"clips": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}
        
        # Extract all clips from all videos
        all_clips = []
        for video in user.get('videos', []):
            for clip in video.get('clips', []):
                clip['video_id'] = video['id']  # Add video_id to clip
                all_clips.append(clip)
        
        total_count = len(all_clips)
        
        # Apply pagination
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_clips = all_clips[start_index:end_index]
        
        # Convert to ClipModel objects
        clip_models = [ClipModel(**clip) for clip in paginated_clips]
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "clips": clip_models,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error getting clips for user {user_id}: {str(e)}")
        return {"clips": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}


async def delete_user_video(user_id: str, video_id: str) -> bool:
    """
    Delete a video for a user.
    
    Args:
        user_id: User ID
        video_id: Video ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_database()
        users_collection = db.users
        
        # Get video to delete S3 files
        video = await get_user_video(user_id, video_id)
        if video:
            # Delete S3 files
            if video.s3_key:
                s3_client.delete_from_s3(video.s3_key)
            
            # Delete clip S3 files
            for clip in video.clips:
                if clip.s3_key:
                    s3_client.delete_from_s3(clip.s3_key)
        
        # Remove video from user's videos array
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$pull": {"videos": {"id": video_id}},
                "$set": {"updated_at": utc_now()}
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Deleted video {video_id} for user {user_id}")
            return True
        else:
            logger.warning(f"No video {video_id} found for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting video {video_id} for user {user_id}: {str(e)}")
        return False 