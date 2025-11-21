"""
Video service for handling user video operations, S3 storage, and YouTube downloads.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from bson import ObjectId
from pymongo.collection import Collection

from database.connection import get_database
from models.video import (
    VideoCreate, VideoUpdate, VideoInDB, Video, 
    ClipCreate, ClipInDB, Clip, VideoType, VideoStatus,
    VideoHistoryResponse, ClipHistoryResponse
)
from utils.s3_storage import s3_client

logger = logging.getLogger(__name__)


class VideoService:
    """Service for video operations"""
    
    def __init__(self):
        self.db = get_database()
        self.videos_collection: Collection = self.db["videos"]
        self.clips_collection: Collection = self.db["clips"]
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes"""
        try:
            # Video indexes
            self.videos_collection.create_index([("user_id", 1)])
            self.videos_collection.create_index([("user_id", 1), ("created_at", -1)])
            self.videos_collection.create_index([("status", 1)])
            self.videos_collection.create_index([("video_type", 1)])
            
            # Clip indexes
            self.clips_collection.create_index([("user_id", 1)])
            self.clips_collection.create_index([("user_id", 1), ("created_at", -1)])
            self.clips_collection.create_index([("video_id", 1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
    
    def _video_to_dict(self, video: VideoInDB) -> Video:
        """Convert VideoInDB to Video response model"""
        return Video(
            id=str(video.id),
            user_id=video.user_id,
            filename=video.filename,
            title=video.title,
            description=video.description,
            duration=video.duration,
            file_size=video.file_size,
            content_type=video.content_type,
            video_type=video.video_type,
            source_url=video.source_url,
            s3_key=video.s3_key,
            s3_url=video.s3_url,
            thumbnail_url=video.thumbnail_url,
            status=video.status,
            error_message=video.error_message,
            created_at=video.created_at,
            updated_at=video.updated_at,
            processed_at=video.processed_at
        )
    
    def _clip_to_dict(self, clip: ClipInDB) -> Clip:
        """Convert ClipInDB to Clip response model"""
        return Clip(
            id=str(clip.id),
            user_id=clip.user_id,
            video_id=clip.video_id,
            title=clip.title,
            description=clip.description,
            start_time=clip.start_time,
            end_time=clip.end_time,
            duration=clip.duration,
            s3_key=clip.s3_key,
            s3_url=clip.s3_url,
            thumbnail_url=clip.thumbnail_url,
            transcription=clip.transcription,
            summary=clip.summary,
            tags=clip.tags,
            metadata=clip.metadata,
            created_at=clip.created_at,
            updated_at=clip.updated_at
        )
    
    async def create_video(self, video_data: VideoCreate) -> Video:
        """Create a new video record"""
        try:
            video = VideoInDB(**video_data.dict())
            result = self.videos_collection.insert_one(video.dict(by_alias=True))
            video.id = result.inserted_id
            
            logger.info(f"Created video {video.id} for user {video.user_id}")
            return self._video_to_dict(video)
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise
    
    async def get_video_by_id(self, video_id: str, user_id: str) -> Optional[Video]:
        """Get video by ID for specific user"""
        try:
            # First try the new user video service structure
            from services.user_video_service import get_user_video
            video = await get_user_video(user_id, video_id)
            if video:
                # Convert VideoModel to Video response format with clips
                # Extract clips and convert ObjectIds to strings
                clips = []
                if video.clips:
                    for clip in video.clips:
                        clip_dict = clip.dict() if hasattr(clip, 'dict') else clip
                        # Convert ObjectId to string if present
                        if 'id' in clip_dict and hasattr(clip_dict['id'], '__str__'):
                            clip_dict['id'] = str(clip_dict['id'])
                        clips.append(clip_dict)
                
                return Video(
                    id=video.id,
                    user_id=user_id,
                    filename=video.filename,
                    title=video.title,
                    description=video.description,
                    duration=video.duration,
                    file_size=video.file_size,
                    content_type=video.content_type,
                    video_type=video.video_type,
                    source_url=video.source_url,
                    s3_key=video.s3_key,
                    s3_url=video.s3_url,
                    thumbnail_url=video.thumbnail_url,
                    status=video.status,
                    error_message=video.error_message,
                    created_at=video.created_at,
                    updated_at=video.updated_at,
                    processed_at=video.processed_at,
                    clips=clips
                )
            
            # Fallback to old structure if not found in new structure
            try:
                video_doc = self.videos_collection.find_one({
                    "_id": ObjectId(video_id),
                    "user_id": user_id
                })
                
                if video_doc:
                    video = VideoInDB(**video_doc)
                    return self._video_to_dict(video)
            except Exception as old_error:
                logger.debug(f"Video not found in old structure: {old_error}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting video {video_id}: {str(e)}")
            return None
    
    async def update_video(self, video_id: str, user_id: str, update_data: VideoUpdate) -> Optional[Video]:
        """Update video record"""
        try:
            # First try the new user video service structure
            from services.user_video_service import update_user_video
            update_dict = update_data.dict(exclude_unset=True)
            success = await update_user_video(user_id, video_id, update_dict)
            
            if success:
                return await self.get_video_by_id(video_id, user_id)
            
            # Fallback to old structure if not found in new structure
            try:
                update_dict["updated_at"] = datetime.now(timezone.utc)
                
                result = self.videos_collection.update_one(
                    {"_id": ObjectId(video_id), "user_id": user_id},
                    {"$set": update_dict}
                )
                
                if result.modified_count > 0:
                    return await self.get_video_by_id(video_id, user_id)
            except Exception as old_error:
                logger.debug(f"Video not found in old structure: {old_error}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating video {video_id}: {str(e)}")
            return None
    
    async def get_user_videos(
        self, 
        user_id: str, 
        page: int = 1, 
        page_size: int = 20,
        video_type: Optional[VideoType] = None,
        status: Optional[VideoStatus] = None
    ) -> VideoHistoryResponse:
        """Get paginated video history for user"""
        try:
            # First try the new user video service structure
            from services.user_video_service import get_user_videos as get_user_videos_new
            result = await get_user_videos_new(user_id, page, page_size)
            
            if result and result.get("videos"):
                # Convert VideoModel objects to Video response format
                videos = []
                for video_model in result["videos"]:
                    # Convert clips to dict format
                    clips_data = []
                    for clip in video_model.clips:
                        clip_dict = {
                            "id": clip.id if hasattr(clip, 'id') else clip.get('id'),
                            "title": clip.title if hasattr(clip, 'title') else clip.get('title'),
                            "start_time": clip.start_time if hasattr(clip, 'start_time') else clip.get('start_time', 0),
                            "end_time": clip.end_time if hasattr(clip, 'end_time') else clip.get('end_time', 0),
                            "duration": clip.duration if hasattr(clip, 'duration') else clip.get('duration', 0),
                            "s3_key": clip.s3_key if hasattr(clip, 's3_key') else clip.get('s3_key'),
                            "s3_url": clip.s3_url if hasattr(clip, 's3_url') else clip.get('s3_url'),
                            "thumbnail_url": clip.thumbnail_url if hasattr(clip, 'thumbnail_url') else clip.get('thumbnail_url'),
                            "transcription": clip.transcription if hasattr(clip, 'transcription') else clip.get('transcription', ''),
                        }
                        clips_data.append(clip_dict)
                    
                    video = Video(
                        id=video_model.id,
                        user_id=user_id,
                        filename=video_model.filename,
                        title=video_model.title,
                        description=video_model.description,
                        duration=video_model.duration,
                        file_size=video_model.file_size,
                        content_type=video_model.content_type,
                        video_type=video_model.video_type,
                        source_url=video_model.source_url,
                        s3_key=video_model.s3_key,
                        s3_url=video_model.s3_url,
                        thumbnail_url=video_model.thumbnail_url,
                        status=video_model.status,
                        error_message=video_model.error_message,
                        created_at=video_model.created_at,
                        updated_at=video_model.updated_at,
                        processed_at=video_model.processed_at,
                        clips=clips_data  # Include clips in response
                    )
                    videos.append(video)
                
                return VideoHistoryResponse(
                    videos=videos,
                    total_count=result["total_count"],
                    page=result["page"],
                    page_size=result["page_size"],
                    total_pages=result["total_pages"]
                )
            
            # Fallback to old structure if no videos found in new structure
            try:
                # Build filter
                filter_query = {"user_id": user_id}
                if video_type:
                    filter_query["video_type"] = video_type
                if status:
                    filter_query["status"] = status
                
                # Get total count
                total_count = self.videos_collection.count_documents(filter_query)
                
                # Calculate pagination
                skip = (page - 1) * page_size
                total_pages = (total_count + page_size - 1) // page_size
                
                # Get videos
                cursor = self.videos_collection.find(filter_query).sort(
                    "created_at", -1
                ).skip(skip).limit(page_size)
                
                videos = []
                for video_doc in cursor:
                    video = VideoInDB(**video_doc)
                    videos.append(self._video_to_dict(video))
                
                return VideoHistoryResponse(
                    videos=videos,
                    total_count=total_count,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages
                )
            except Exception as old_error:
                logger.debug(f"No videos found in old structure: {old_error}")
                return VideoHistoryResponse(
                    videos=[],
                    total_count=0,
                    page=page,
                    page_size=page_size,
                    total_pages=0
                )
            
        except Exception as e:
            logger.error(f"Error getting videos for user {user_id}: {str(e)}")
            raise
    
    async def create_clip(self, clip_data: ClipCreate) -> Clip:
        """Create a new clip record"""
        try:
            clip = ClipInDB(**clip_data.dict())
            result = self.clips_collection.insert_one(clip.dict(by_alias=True))
            clip.id = result.inserted_id
            
            logger.info(f"Created clip {clip.id} for user {clip.user_id}")
            return self._clip_to_dict(clip)
            
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}")
            raise
    
    async def get_user_clips(
        self, 
        user_id: str, 
        page: int = 1, 
        page_size: int = 20,
        video_id: Optional[str] = None
    ) -> ClipHistoryResponse:
        """Get paginated clip history for user"""
        try:
            # First try the new user video service structure
            from services.user_video_service import get_user_clips as get_user_clips_new
            result = await get_user_clips_new(user_id, page, page_size)
            
            if result and result.get("clips"):
                # Convert ClipModel objects to Clip response format
                clips = []
                for clip_model in result["clips"]:
                    clip = Clip(
                        id=clip_model.id,
                        user_id=user_id,
                        video_id=clip_model.video_id if hasattr(clip_model, 'video_id') else video_id,
                        title=clip_model.title,
                        description=clip_model.description,
                        start_time=clip_model.start_time,
                        end_time=clip_model.end_time,
                        duration=clip_model.duration,
                        s3_key=clip_model.s3_key,
                        s3_url=clip_model.s3_url,
                        thumbnail_url=clip_model.thumbnail_url,
                        transcription=clip_model.transcription,
                        summary=clip_model.summary,
                        tags=clip_model.tags,
                        metadata=clip_model.metadata,
                        created_at=clip_model.created_at,
                        updated_at=clip_model.updated_at
                    )
                    clips.append(clip)
                
                return ClipHistoryResponse(
                    clips=clips,
                    total_count=result["total_count"],
                    page=result["page"],
                    page_size=result["page_size"],
                    total_pages=result["total_pages"]
                )
            
            # Fallback to old structure if no clips found in new structure
            try:
                # Build filter
                filter_query = {"user_id": user_id}
                if video_id:
                    filter_query["video_id"] = video_id
                
                # Get total count
                total_count = self.clips_collection.count_documents(filter_query)
                
                # Calculate pagination
                skip = (page - 1) * page_size
                total_pages = (total_count + page_size - 1) // page_size
                
                # Get clips
                cursor = self.clips_collection.find(filter_query).sort(
                    "created_at", -1
                ).skip(skip).limit(page_size)
                
                clips = []
                for clip_doc in cursor:
                    clip = ClipInDB(**clip_doc)
                    clips.append(self._clip_to_dict(clip))
                
                return ClipHistoryResponse(
                    clips=clips,
                    total_count=total_count,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages
                )
            except Exception as old_error:
                logger.debug(f"No clips found in old structure: {old_error}")
                return ClipHistoryResponse(
                    clips=[],
                    total_count=0,
                    page=page,
                    page_size=page_size,
                    total_pages=0
                )
            
        except Exception as e:
            logger.error(f"Error getting clips for user {user_id}: {str(e)}")
            raise

    async def get_user_videos_with_clips(
        self, 
        user_id: str, 
        page: int = 1, 
        page_size: int = 20,
        include_clips: bool = True
    ) -> Dict[str, Any]:
        """
        Get user's videos with their associated clips efficiently.
        
        Returns:
        {
            "videos": [
                {
                    "id": "video_id",
                    "title": "Video Title", 
                    "s3_url": "video_url",
                    "clips": [
                        {
                            "id": "clip_id",
                            "title": "Clip Title",
                            "s3_url": "clip_url",
                            "start_time": 10.5,
                            "end_time": 25.0
                        }
                    ]
                }
            ],
            "total_count": 10,
            "page": 1,
            "total_pages": 1
        }
        """
        try:
            # Get user's videos
            videos_response = await self.get_user_videos(
                user_id=user_id,
                page=page,
                page_size=page_size
            )
            
            if not include_clips:
                return {
                    "videos": videos_response.videos,
                    "total_count": videos_response.total_count,
                    "page": videos_response.page,
                    "total_pages": videos_response.total_pages
                }
            
            # For each video, get its clips
            videos_with_clips = []
            for video in videos_response.videos:
                # Get clips for this video
                clips_response = await self.get_user_clips(
                    user_id=user_id,
                    video_id=video.id,
                    page=1,
                    page_size=100  # Get all clips for this video
                )
                
                video_data = {
                    "id": video.id,
                    "title": video.title,
                    "s3_url": video.s3_url,
                    "thumbnail_url": video.thumbnail_url,
                    "duration": video.duration,
                    "created_at": video.created_at,
                    "clips": clips_response.clips
                }
                videos_with_clips.append(video_data)
            
            return {
                "videos": videos_with_clips,
                "total_count": videos_response.total_count,
                "page": videos_response.page,
                "total_pages": videos_response.total_pages
            }
            
        except Exception as e:
            logger.error(f"Error getting videos with clips for user {user_id}: {str(e)}")
            raise
    
    def generate_user_s3_key(self, user_id: str, filename: str, video_type: VideoType = VideoType.UPLOAD) -> str:
        """Generate S3 key for user-specific storage"""
        video_id = str(uuid.uuid4())
        prefix = "youtube" if video_type == VideoType.YOUTUBE else "uploads"
        return f"users/{user_id}/{prefix}/{video_id}/{filename}"
    
    def generate_clip_s3_key(self, user_id: str, video_id: str, clip_id: str, filename: str) -> str:
        """Generate S3 key for clip storage"""
        return f"users/{user_id}/clips/{video_id}/{clip_id}/{filename}"
    
    async def get_s3_upload_url(self, user_id: str, filename: str, content_type: str = "video/mp4") -> Optional[Dict[str, Any]]:
        """Generate S3 presigned upload URL for user"""
        try:
            if not s3_client.available:
                logger.error("S3 client not available")
                return None
            
            video_id = str(uuid.uuid4())
            s3_key = self.generate_user_s3_key(user_id, filename, VideoType.UPLOAD)
            
            presigned_data = s3_client.generate_presigned_post(
                video_id=video_id,
                filename=filename,
                content_type=content_type
            )
            
            if presigned_data:
                # Override the S3 key to use user-specific path
                presigned_data["s3_key"] = s3_key
                presigned_data["video_id"] = video_id
                return presigned_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating S3 upload URL: {str(e)}")
            return None
    
    async def get_s3_download_url(self, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate S3 presigned download URL"""
        try:
            if not s3_client.available:
                return None
            
            return s3_client.generate_presigned_url(s3_key, expires_in)
            
        except Exception as e:
            logger.error(f"Error generating S3 download URL: {str(e)}")
            return None
    
    async def delete_video(self, video_id: str, user_id: str) -> bool:
        """Delete video and its associated clips"""
        try:
            # First try the new user video service structure
            from services.user_video_service import delete_user_video
            success = await delete_user_video(user_id, video_id)
            
            if success:
                return True
            
            # Fallback to old structure if not found in new structure
            try:
                # Delete clips first
                self.clips_collection.delete_many({
                    "video_id": video_id,
                    "user_id": user_id
                })
                
                # Delete video
                result = self.videos_collection.delete_one({
                    "_id": ObjectId(video_id),
                    "user_id": user_id
                })
                
                return result.deleted_count > 0
            except Exception as old_error:
                logger.debug(f"Video not found in old structure: {old_error}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting video {video_id}: {str(e)}")
            return False


# Global video service instance
video_service = VideoService() 