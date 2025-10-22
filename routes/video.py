"""
Video routes for user-specific video operations, S3 uploads, and YouTube downloads.
"""

import logging
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import HttpUrl
from bson import ObjectId

from models.user import User
from models.video import (
    Video, Clip, VideoHistoryResponse, ClipHistoryResponse,
    VideoType, VideoStatus, VideoCreate, VideoUpdate
)
from services.auth import auth_service
from services.video_service import video_service
# from utils.sieve_downloader import download_youtube_video_sieve  # Temporarily disabled

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/videos", tags=["Videos"])

# Security scheme
security = HTTPBearer()


async def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = auth_service.verify_token(credentials.credentials, "access")
    if token_data is None or token_data.user_id is None:
        raise credentials_exception
    
    user = await auth_service.get_user_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return auth_service.user_to_dict(user)


@router.get("/upload-url")
async def get_s3_upload_url(
    current_user: Annotated[User, Depends(get_current_user)],
    filename: str = Query(..., description="Name of the file to upload"),
    content_type: str = Query("video/mp4", description="Content type of the file")
):
    """
    Get S3 presigned upload URL for user-specific video upload.
    
    - **filename**: Name of the file to upload
    - **content_type**: Content type (default: video/mp4)
    
    Returns presigned URL and fields for direct S3 upload.
    """
    try:
        presigned_data = await video_service.get_s3_upload_url(
            user_id=current_user.id,
            filename=filename,
            content_type=content_type
        )
        
        if not presigned_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate upload URL"
            )
        
        return {
            "upload_url": presigned_data["url"],
            "fields": presigned_data["fields"],
            "s3_key": presigned_data["s3_key"],
            "video_id": presigned_data["video_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating upload URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/register-upload")
async def register_uploaded_video(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str = Form(..., description="Video ID from upload URL"),
    s3_key: str = Form(..., description="S3 key where video was uploaded"),
    filename: str = Form(..., description="Original filename"),
    title: Optional[str] = Form(None, description="Video title"),
    description: Optional[str] = Form(None, description="Video description")
):
    """
    Register a video that was uploaded to S3.
    """
    try:
        # Create video record
        video_data = VideoCreate(
            user_id=current_user.id,
            filename=filename,
            title=title,
            description=description,
            s3_key=s3_key,
            status=VideoStatus.UPLOADING,
            video_type=VideoType.UPLOAD
        )
        # Insert video and fetch the document with _id
        db = video_service.db
        videos_collection = db["videos"]
        video_dict = video_data.dict()
        result = videos_collection.insert_one(video_dict)
        video = videos_collection.find_one({"_id": result.inserted_id})
        # Convert ObjectId to string
        video["_id"] = str(video["_id"])
        video["video_id"] = video["_id"]
        # S3 URL assertion
        if video.get('s3_url') and not str(video['s3_url']).startswith('https://'):
            raise HTTPException(status_code=500, detail='Non-S3 URL detected in video object!')
        return {
            "message": "Video registered successfully",
            "video": video
        }
    except Exception as e:
        logger.error(f"Error registering uploaded video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/youtube-download")
async def download_youtube_video(
    current_user: Annotated[User, Depends(get_current_user)],
    url: HttpUrl = Query(..., description="YouTube video URL"),
    title: Optional[str] = Query(None, description="Custom title for the video"),
    description: Optional[str] = Query(None, description="Video description")
):
    """
    Register YouTube video for download and storage in S3 for the user.
    """
    try:
        # Extract video ID from URL
        video_id_from_url = url.path.split('/')[-1] if url.path else "unknown"
        # Create initial video record
        video_data = VideoCreate(
            user_id=current_user.id,
            filename=f"youtube_{video_id_from_url}.mp4",
            title=title or f"YouTube Video {video_id_from_url}",
            description=description,
            source_url=str(url),
            status=VideoStatus.DOWNLOADING,
            video_type=VideoType.YOUTUBE
        )
        # Insert video and fetch the document with _id
        db = video_service.db
        videos_collection = db["videos"]
        video_dict = video_data.dict()
        result = videos_collection.insert_one(video_dict)
        video = videos_collection.find_one({"_id": result.inserted_id})
        # Convert ObjectId to string
        video["_id"] = str(video["_id"])
        video["video_id"] = video["_id"]
        # S3 URL assertion
        if video.get('s3_url') and not str(video['s3_url']).startswith('https://'):
            raise HTTPException(status_code=500, detail='Non-S3 URL detected in video object!')
        return {
            "message": "YouTube video registered for download",
            "video": video,
            "note": "Actual download functionality will be implemented in a future update"
        }
    except Exception as e:
        logger.error(f"Error registering YouTube video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/history")
async def get_video_history(
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    video_type: Optional[VideoType] = Query(None, description="Filter by video type"),
    status: Optional[VideoStatus] = Query(None, description="Filter by status")
):
    """
    Get user's video upload and download history.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **video_type**: Filter by video type (upload/youtube)
    - **status**: Filter by processing status
    
    Returns paginated list of user's videos.
    """
    try:
        history = await video_service.get_user_videos(
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            video_type=video_type,
            status=status
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting video history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/history/with-clips")
async def get_video_history_with_clips(
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    db = video_service.db
    videos_collection = db["videos"]
    clips_collection = db["clips"]

    videos_cursor = videos_collection.find({"user_id": current_user.id}).skip((page-1)*page_size).limit(page_size)
    videos = []
    for video in videos_cursor:
        video_id = str(video["_id"])
        # Convert all ObjectId fields in video to str
        video = {k: (str(v) if isinstance(v, ObjectId) else v) for k, v in video.items()}
        # Fetch clips for this video
        clips = list(clips_collection.find({"video_id": video_id}))
        video["clips"] = [
            {
                "clip_id": str(clip["_id"]),
                "title": clip.get("title"),
                "s3_url": clip.get("s3_url"),
                "thumbnail_url": clip.get("thumbnail_url"),
                "start_time": clip.get("start_time"),
                "end_time": clip.get("end_time"),
                # Convert ObjectId fields in clip to str
                **{k: (str(v) if isinstance(v, ObjectId) else v) for k, v in clip.items() if k not in ["_id"]}
            }
            for clip in clips
        ]
        video["video_id"] = video_id
        videos.append(video)
    total_count = videos_collection.count_documents({"user_id": current_user.id})
    total_pages = (total_count + page_size - 1) // page_size
    return {
        "user_id": current_user.id,
        "videos": videos,
        "total_count": total_count,
        "page": page,
        "total_pages": total_pages
    }


@router.get("/{video_id}")
async def get_video(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str
):
    """
    Get specific video by ID.
    
    - **video_id**: Video ID
    
    Returns video details if owned by the user.
    """
    try:
        logger.info(f"[get_video] User {current_user.id} requesting video {video_id}")
        video = await video_service.get_video_by_id(video_id, current_user.id)
        if not video:
            logger.warning(f"[get_video] Video {video_id} not found or not owned by user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found or not owned by user"
            )
        # S3 URL assertion for get_video
        if (hasattr(video, 's3_url') and video.s3_url and not str(video.s3_url).startswith('https://')) or (isinstance(video, dict) and video.get('s3_url') and not str(video['s3_url']).startswith('https://')):
            logger.error(f"[get_video] Non-S3 URL detected for video {video_id} (user {current_user.id})")
            raise HTTPException(status_code=500, detail='Non-S3 URL detected in video object!')
        logger.info(f"[get_video] Returning video {video_id} for user {current_user.id}")
        return video
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/{video_id}")
async def update_video(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str,
    update_data: VideoUpdate
):
    """
    Update video metadata.
    
    - **video_id**: Video ID
    - **update_data**: Video update data
    
    Returns updated video details.
    """
    try:
        video = await video_service.update_video(video_id, current_user.id, update_data)
        
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        return video
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/{video_id}")
async def delete_video(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str
):
    """
    Delete video and its associated clips.
    
    - **video_id**: Video ID
    
    Returns success message.
    """
    try:
        success = await video_service.delete_video(video_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        return {"message": "Video deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/{video_id}/download-url")
async def get_video_download_url(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str,
    expires_in: int = Query(3600, ge=300, le=86400, description="URL expiration time in seconds")
):
    """
    Get presigned download URL for video.
    
    - **video_id**: Video ID
    - **expires_in**: URL expiration time in seconds (default: 1 hour, max: 24 hours)
    
    Returns presigned download URL.
    """
    try:
        video = await video_service.get_video_by_id(video_id, current_user.id)
        
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        if not video.s3_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video not available for download"
            )
        
        download_url = await video_service.get_s3_download_url(video.s3_key, expires_in)
        
        if not download_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate download URL"
            )
        
        return {
            "download_url": download_url,
            "expires_in": expires_in
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/clips/history")
async def get_clip_history(
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    video_id: Optional[str] = Query(None, description="Filter by video ID")
):
    """
    Get user's clip history.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 20, max: 100)
    - **video_id**: Filter by video ID
    
    Returns paginated list of user's clips.
    """
    try:
        history = await video_service.get_user_clips(
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            video_id=video_id
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting clip history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        ) 