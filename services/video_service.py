"""
Service layer for Video operations - Class-based
"""
import logging
from typing import Optional, List, Dict, Any
from bson import ObjectId
from database.connection import get_database
from models.video import Video, VideoStatus, VideoType

logger = logging.getLogger(__name__)


class VideoService:
    """Video service for managing videos in separate collection"""
    
    def __init__(self):
        self.db = get_database()
        self.videos_collection = self.db.videos
        self.clips_collection = self.db.clips
    
    async def create_video(self, user_id: str, video_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new video in videos collection
        
        Args:
            user_id: User ID who owns the video
            video_data: Video data dictionary
            
        Returns:
            str: Video ID if successful, None otherwise
        """
        try:
            # Create video model
            video = Video(
                user_id=user_id,
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
                process_task_id=video_data.get('process_task_id')
            )
            
            # Insert into collection
            video_dict = video.dict(by_alias=True)
            result = self.videos_collection.insert_one(video_dict)
            
            logger.info(f"✅ Created video {video.id} for user {user_id}")
            return video.id
            
        except Exception as e:
            logger.error(f"❌ Error creating video for user {user_id}: {str(e)}")
            return None
    
    async def get_video(self, video_id: str) -> Optional[Video]:
        """Get a video by ID"""
        try:
            video_data = self.videos_collection.find_one({"_id": video_id})
            
            if video_data:
                return Video(**video_data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting video {video_id}: {str(e)}")
            return None
    
    async def get_video_by_id(self, video_id: str, user_id: str) -> Optional[Video]:
        """Get a video by ID for a specific user"""
        try:
            video_data = self.videos_collection.find_one({"_id": video_id, "user_id": user_id})
            
            if video_data:
                return Video(**video_data)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting video {video_id}: {str(e)}")
            return None
    
    async def get_user_videos(self, user_id: str, page: int = 1, page_size: int = 10, **filters) -> Dict[str, Any]:
        """Get all videos for a user with pagination"""
        try:
            skip = (page - 1) * page_size
            query = {"user_id": user_id}
            
            # Add filters if provided
            if filters.get('video_type'):
                query['video_type'] = filters['video_type']
            if filters.get('status'):
                query['status'] = filters['status']
            
            cursor = self.videos_collection.find(query).sort("created_at", -1).skip(skip).limit(page_size)
            
            videos = [Video(**video_data) for video_data in cursor]
            total_count = self.videos_collection.count_documents(query)
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "user_id": user_id,
                "videos": videos,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting videos for user {user_id}: {str(e)}")
            return {"videos": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}
    
    async def update_video(self, video_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Video]:
        """Update a video"""
        try:
            from models.video import utc_now
            updates['updated_at'] = utc_now()
            
            result = self.videos_collection.update_one(
                {"_id": video_id, "user_id": user_id},
                {"$set": updates}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated video {video_id}")
                return await self.get_video(video_id)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error updating video {video_id}: {str(e)}")
            return None
    
    async def delete_video(self, video_id: str, user_id: str) -> bool:
        """Delete a video and its clips"""
        try:
            # Delete associated clips first
            self.clips_collection.delete_many({"video_id": video_id})
            
            # Delete video
            result = self.videos_collection.delete_one({"_id": video_id, "user_id": user_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Deleted video {video_id} and its clips")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error deleting video {video_id}: {str(e)}")
            return False
    
    async def get_user_clips(self, user_id: str, page: int = 1, page_size: int = 20, video_id: Optional[str] = None) -> Dict[str, Any]:
        """Get clips for a user"""
        try:
            skip = (page - 1) * page_size
            query = {"user_id": user_id}
            
            if video_id:
                query['video_id'] = video_id
            
            cursor = self.clips_collection.find(query).sort("created_at", -1).skip(skip).limit(page_size)
            
            from models.clip import Clip
            clips = [Clip(**clip_data) for clip_data in cursor]
            total_count = self.clips_collection.count_documents(query)
            total_pages = (total_count + page_size - 1) // page_size
            
            return {
                "user_id": user_id,
                "clips": clips,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting clips for user {user_id}: {str(e)}")
            return {"clips": [], "total_count": 0, "page": page, "page_size": page_size, "total_pages": 0}
    
    async def get_s3_upload_url(self, user_id: str, filename: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Generate S3 presigned upload URL - placeholder for now"""
        # TODO: Implement S3 upload URL generation
        logger.warning("get_s3_upload_url not yet implemented in new service")
        return None
    
    async def get_s3_download_url(self, s3_key: str, expires_in: int) -> Optional[str]:
        """Generate S3 presigned download URL - placeholder for now"""
        # TODO: Implement S3 download URL generation
        logger.warning("get_s3_download_url not yet implemented in new service")
        return None


# Create singleton instance
video_service = VideoService()
