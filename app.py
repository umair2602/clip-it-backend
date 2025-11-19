import asyncio
import datetime
import json
import tempfile

# Set up logging
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, HttpUrl

# Import logging configuration
from logging_config import setup_logging

# Set up logging to file and console
setup_logging(log_dir="logs", log_file="backend.log", log_level=logging.INFO)

# Import configuration
from config import settings
from services.user_video_service import utc_now
from services.content_analysis import analyze_content
from services.face_tracking import track_faces

# Import services
from services.transcription import load_model, transcribe_audio, transcribe_audio_sync
from services.video_processing import create_clip, generate_thumbnail, process_video
from services.user_video_service import create_user_video, update_video_s3_url, get_user_video, get_user_videos, get_user_clips, add_clip_to_video

# Import S3 client and validator
from utils.s3_storage import s3_client
from utils.s3_storage_multipart import multipart_s3_client  # For optimized multipart uploads
from utils.s3_validator import assert_s3_url, validate_response_urls, sanitize_urls_for_response, assert_response_urls
from utils.sieve_downloader import download_youtube_video_sieve

# Import YouTube downloader (both original and new Sieve version)
from utils.youtube_downloader import download_youtube_video

# Import authentication routes
from routes.auth import router as auth_router, get_current_user

# Import video routes
from routes.video import router as video_router
from routes.tiktok import router as tiktok_router

# Import models
from models.user import User, VideoType, VideoStatus

# Import database connection
from database.connection import mongodb, get_database

from bson import ObjectId

import cv2

def generate_video_thumbnail(video_path, thumbnail_path, timestamp=1.0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumbnail_path, frame)
        cap.release()
        return True
    cap.release()
    return False


# ============================================================================
# PARALLEL CLIP PROCESSING FUNCTIONS
# ============================================================================

async def process_single_clip_async(
    clip: dict,
    index: int,
    file_path: str,
    video_id: str,
    output_dir: str,
    s3_client_instance
) -> tuple:
    """
    Process a single clip completely (creation + thumbnail + upload).
    
    Returns tuple of (index, result_dict) for result tracking
    """
    try:
        logging.info(f"[PARALLEL] Starting async processing of clip {index}: {clip.get('id')}")
        
        # Step 1: Create the clip
        clip_path = await create_clip(
            video_path=file_path,
            output_dir=output_dir,
            start_time=clip["start_time"],
            end_time=clip["end_time"],
            clip_id=clip.get("id"),
        )
        
        if not clip_path:
            logging.warning(f"[PARALLEL] Failed to create clip {index}")
            return index, None
        
        logging.info(f"[PARALLEL] Clip {index} created at {clip_path}")
        
        # Step 2: Generate thumbnail
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_thumb:
            thumbnail_path = temp_thumb.name
        
        clip_duration = clip.get("end_time", 0) - clip.get("start_time", 0)
        thumbnail_timestamp = (
            min(clip_duration / 3, clip_duration - 0.5)
            if clip_duration > 1.5
            else 0
        )
        
        generated_path = await generate_thumbnail(
            clip_path, thumbnail_path, thumbnail_timestamp
        )
        
        logging.info(f"[PARALLEL] Clip {index} thumbnail generated: {generated_path}")
        
        # Step 3: Upload to S3 (clip and thumbnail in parallel)
        clip_upload = asyncio.create_task(
            _upload_clip_to_s3_async(clip_path, video_id, clip.get("id"), s3_client_instance)
        )
        
        thumbnail_upload = None
        if generated_path and os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
            thumbnail_upload = asyncio.create_task(
                _upload_thumbnail_to_s3_async(thumbnail_path, video_id, clip.get("id"), s3_client_instance)
            )
        
        # Wait for both uploads
        clip_result = await clip_upload
        thumb_result = await thumbnail_upload if thumbnail_upload else (False, None)
        
        logging.info(f"[PARALLEL] Clip {index} upload: {clip_result[0]}, Thumbnail upload: {thumb_result[0]}")
        
        # Step 4: Cleanup
        try:
            if os.path.exists(clip_path):
                os.unlink(clip_path)
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.unlink(thumbnail_path)
        except Exception as cleanup_err:
            logging.warning(f"[PARALLEL] Cleanup error for clip {index}: {cleanup_err}")
        
        # Return results
        return index, {
            'clip_s3_key': clip_result[1],
            'thumbnail_s3_key': thumb_result[1],
            'title': clip.get('title'),
            'start_time': clip.get('start_time'),
            'end_time': clip.get('end_time'),
        }
        
    except Exception as e:
        logging.error(f"[PARALLEL] Error processing clip {index}: {str(e)}")
        import traceback
        logging.error(f"[PARALLEL] Traceback: {traceback.format_exc()}")
        return index, None


async def _upload_clip_to_s3_async(clip_path: str, video_id: str, clip_id: str, s3_client_instance) -> tuple:
    """Wrapper for async S3 clip upload (runs in thread pool)
    
    Automatically uses multipart upload for large files (>50MB)
    """
    loop = asyncio.get_event_loop()
    
    # Check file size to decide upload method
    file_size = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
    
    # Use multipart for files > 50MB, regular for smaller
    if file_size > 50 * 1024 * 1024:
        logging.info(f"[PARALLEL] File size {file_size / 1024 / 1024:.1f}MB - using optimized multipart upload")
        return await loop.run_in_executor(
            None,
            lambda: multipart_s3_client.upload_clip_to_s3_optimized(clip_path, video_id, clip_id)
        )
    else:
        logging.info(f"[PARALLEL] File size {file_size / 1024 / 1024:.1f}MB - using standard upload")
        return await loop.run_in_executor(
            None,
            lambda: s3_client_instance.upload_clip_to_s3(clip_path, video_id, clip_id)
        )


async def _upload_thumbnail_to_s3_async(thumb_path: str, video_id: str, clip_id: str, s3_client_instance) -> tuple:
    """Wrapper for async S3 thumbnail upload (runs in thread pool)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: s3_client_instance.upload_thumbnail_to_s3(thumb_path, video_id, clip_id)
    )

# ============================================================================
# END PARALLEL CLIP PROCESSING FUNCTIONS
# ============================================================================

# Logging is now configured in logging_config.py (imported above)
# The old basicConfig has been removed to avoid conflicts

# Create FastAPI app
app = FastAPI(
    title="AI Podcast Clipper API",
    description="API for AI-powered podcast clipping with user authentication",
    version="1.0.0"
)

# Include authentication routes
app.include_router(auth_router)

# Include video routes
app.include_router(video_router)

# Include TikTok routes
app.include_router(tiktok_router)

# TikTok site verification
VERIFICATION_FILENAME = "tiktok4QPEpS6YFmd4DmFfdr2Kjw4YKsWvEWky.txt"

@app.get(f"/{VERIFICATION_FILENAME}", include_in_schema=False)
def serve_tiktok_verification():
    # Must be 200, no redirects, text/plain
    verification_path = Path(__file__).parent / "verify" / VERIFICATION_FILENAME
    print(f"DEBUG: Verification file path: {verification_path}")
    print(f"DEBUG: File exists: {verification_path.exists()}")
    if not verification_path.exists():
        raise HTTPException(status_code=404, detail=f"Verification file not found at {verification_path}")
    return FileResponse(verification_path, media_type="text/plain")



# Pre-load the Whisper model
whisper_model = None


@app.on_event("startup")
async def startup_event():
    global whisper_model

    # Initialize MongoDB connection
    try:
        logging.info("Connecting to MongoDB...")
        if mongodb.connect():
            logging.info("MongoDB connected successfully")
        else:
            logging.error("Failed to connect to MongoDB")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {str(e)}")

    # Pre-load Whisper model
    try:
        logging.info("Pre-loading Whisper model...")
        # Check for GPU availability
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            try:
                logging.info(f"CUDA version: {torch.version.cuda}")
            except AttributeError:
                logging.info("CUDA version information not available")
            logging.info(
                f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            logging.warning(
                "No GPU detected. Using CPU for transcription (will be slower)"
            )

        whisper_model = load_model(model_size="tiny")
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Whisper model: {str(e)}", exc_info=True)
        # We'll continue without pre-loading and let the transcribe_audio function
        # handle model loading when needed


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload and output directories if they don't exist
upload_dir = Path("uploads")
output_dir = Path("outputs")
static_dir = Path("static")  # Add static directory for default assets
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

# Create default thumbnail if it doesn't exist
default_thumbnail_path = static_dir / "default_thumbnail.jpg"
if not default_thumbnail_path.exists():
    try:
        # Create a simple default thumbnail - a colored rectangle with text
        img = Image.new("RGB", (640, 360), color=(109, 40, 217))  # Brand purple color
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            font = ImageFont.load_default()
        d.text((220, 160), "No Thumbnail", fill=(255, 255, 255), font=font)
        img.save(default_thumbnail_path)
        logging.info(f"Created default thumbnail at {default_thumbnail_path}")
    except Exception as e:
        logging.error(f"Failed to create default thumbnail: {str(e)}")

# Mount static files for serving uploads and outputs
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Models
class ClipRequest(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    title: Optional[str] = None


class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None


class S3UploadRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


# Add new model for YouTube URL
class YouTubeRequest(BaseModel):
    url: HttpUrl
    auto_process: bool = True


# In-memory storage for task status 
# NOTE: This is deprecated and kept only for backwards compatibility.
# Use MongoDB video status via /status/{video_id} endpoint instead.
# This dictionary is not shared across containers and will be removed in future versions.
tasks = {}

# Tiktok
@app.get(f"/tiktok{settings.TIKTOK_VERIFICATION_KEY}.txt")
async def verify_tiktok_domain():
    return Response(
        content=f"tiktok-developers-site-verification={settings.TIKTOK_VERIFICATION_KEY}",
        media_type="text/plain",
    )

# Routes
@app.get("/")
async def read_root(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None
):
    """Root endpoint that handles TikTok OAuth callbacks since TikTok only verifies the base domain"""
    
    # If this is a TikTok OAuth callback, handle it
    if code or state or error:
        # Handle the OAuth callback
        if error:
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head><title>TikTok OAuth Error</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff4444;">❌ TikTok Authorization Failed</h1>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Description:</strong> {error_description or 'No description provided'}</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)
        
        if code and state:
            # Redirect to the actual TikTok callback endpoint with the parameters
            from fastapi.responses import RedirectResponse
            callback_url = f"/tiktok/callback/?code={code}&state={state}"
            return RedirectResponse(url=callback_url)
    
    # Default response for non-OAuth requests
    return {"message": "Welcome to AI Podcast Clipper API"}


@app.get("/auth")
async def auth_endpoint(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None
):
    """Auth endpoint to handle TikTok OAuth callbacks"""
    
    # If this is a TikTok OAuth callback, handle it
    if code or state or error:
        # Handle the OAuth callback
        if error:
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head><title>TikTok OAuth Error</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff4444;">❌ TikTok Authorization Failed</h1>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Description:</strong> {error_description or 'No description provided'}</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)
        
        if code and state:
            # Redirect to the actual TikTok callback endpoint with the parameters
            from fastapi.responses import RedirectResponse
            callback_url = f"/tiktok/callback?code={code}&state={state}"
            return RedirectResponse(url=callback_url)
    
    # Default response for non-OAuth requests
    return {"message": "Auth endpoint - TikTok OAuth callback handler"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "ok", "whisper_model_loaded": whisper_model is not None}


@app.get("/heartbeat")
async def heartbeat():
    """Enhanced heartbeat endpoint that provides system status information
    and acts as a lightweight ping even when the server is busy
    """
    # Get current active tasks count
    active_tasks = len(
        [
            t
            for t in tasks.values()
            if t.get("status") == "processing"
            or t.get("status") == "transcribing"
            or t.get("status") == "analyzing"
        ]
    )

    # Get completed tasks count
    completed_tasks = len([t for t in tasks.values() if t.get("status") == "completed"])

    # Get failed tasks count
    failed_tasks = len([t for t in tasks.values() if t.get("status") == "error"])

    # Calculate average progress of active tasks
    active_task_progress = [
        t.get("progress", 0)
        for t in tasks.values()
        if t.get("status") in ["processing", "transcribing", "analyzing"]
    ]
    avg_progress = sum(active_task_progress) / max(len(active_task_progress), 1)

    # Basic system info
    import psutil

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
    except ImportError:
        cpu_percent = None
        memory_percent = None

    return {
        "status": "ok",
        "timestamp": str(datetime.datetime.now()),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
        },
        "tasks": {
            "active": active_tasks,
            "completed": completed_tasks,
            "failed": failed_tasks,
            "avg_progress": round(avg_progress, 2) if active_tasks > 0 else 0,
        },
    }


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks, 
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(...)
):
    # Generate a unique ID for the video
    video_id = str(uuid.uuid4())

    # Create a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        # Save the uploaded file to temporary location
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Upload to S3
        success, s3_key = s3_client.upload_file_to_s3(
            temp_file_path, video_id, file.filename, file.content_type
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload file to S3")

        # Generate S3 URL
        s3_url = s3_client.get_object_url(s3_key)
        
        # Validate S3 URL
        assert_s3_url(s3_url, "upload endpoint")

        logging.info(f"Video uploaded successfully to S3: {s3_url}")

        # Generate thumbnail after successful upload
        thumbnail_path = tempfile.mktemp(suffix='.jpg')
        if generate_video_thumbnail(temp_file_path, thumbnail_path):
            thumb_success, thumb_s3_key = s3_client.upload_thumbnail_to_s3(thumbnail_path, video_id)
            if thumb_success:
                thumbnail_url = s3_client.get_object_url(thumb_s3_key)
            else:
                thumbnail_url = None
            try:
                os.unlink(thumbnail_path)
            except:
                pass
        else:
            thumbnail_url = None

        # Create video document in user's videos array
        video_data = {
            "title": file.filename,
            "filename": file.filename,
            "content_type": file.content_type,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "status": VideoStatus.PROCESSING,
            "video_type": VideoType.UPLOAD,
            "thumbnail_url": thumbnail_url,
            "clip_thumbnail_url": thumbnail_url # Initialize clip_thumbnail_url
        }
        
        # Create video in user's videos array
        created_video_id = await create_user_video(current_user.id, video_data)
        
        if not created_video_id:
            logging.error(f"Failed to create video document for user {current_user.id}")
            # Don't fail the upload if database save fails

        # Create a task for processing
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting video processing",
            "video_id": created_video_id or video_id,
            "s3_key": s3_key,
            "s3_url": s3_url,
        }

        # Start background processing immediately with S3
        background_tasks.add_task(
            process_s3_podcast, task_id=task_id, video_id=created_video_id or video_id, s3_key=s3_key, local_path=None, user_id=current_user.id
        )

        # Return the video ID and task ID with S3 information
        response = {
            "video_id": created_video_id or video_id,
            "filename": file.filename,
            "status": "processing",
            "task_id": task_id,
            "s3_url": s3_url
        }
        
        # Validate response URLs
        assert_response_urls(response, "upload endpoint response")
        
        return response

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get task status from MongoDB video document (task_id can be video_id or actual task_id)"""
    try:
        # Import user video service
        from services.user_video_service import get_user_video_by_video_id
        
        # Try to find video by video_id (task_id is often the same as video_id)
        video_info = await get_user_video_by_video_id(task_id)
        
        if not video_info:
            # Fallback: Check in-memory tasks for backwards compatibility
            if task_id in tasks:
                return tasks[task_id]
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Convert video status to task status format
        status_map = {
            "downloading": "downloading",
            "processing": "processing",
            "completed": "completed",
            "failed": "error",
            "uploading": "downloading"
        }
        
        task_status = {
            "status": status_map.get(video_info.get("status", "downloading"), "downloading"),
            "progress": get_progress_from_status(video_info.get("status")),
            "message": get_message_from_status(video_info.get("status"), video_info.get("error_message")),
            "video_id": video_info.get("id"),
            "updated_at": video_info.get("updated_at", datetime.datetime.now()).isoformat() if isinstance(video_info.get("updated_at"), datetime.datetime) else str(video_info.get("updated_at")),
            "created_at": video_info.get("created_at", datetime.datetime.now()).isoformat() if isinstance(video_info.get("created_at"), datetime.datetime) else str(video_info.get("created_at")),
        }
        
        # Add clips if completed
        if video_info.get("status") == "completed" and video_info.get("clips"):
            task_status["clips"] = video_info.get("clips")
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def get_progress_from_status(status: str) -> int:
    """Map video status to progress percentage - aligned with actual pipeline steps"""
    progress_map = {
        # Step 1: Downloading YouTube video
        "queued": 5,                    # Job queued
        "downloading": 10,              # Downloading YouTube video
        "downloaded": 15,               # Download complete
        "uploading": 18,                # Uploading to storage
        
        # Step 2: Transcribing
        "processing_started": 20,       # Starting processing
        "transcribing": 30,             # Transcribing audio with Whisper
        
        # Step 3: Finding interesting content
        "analyzing": 50,                # AI analyzing content for best moments
        
        # Step 4: Generating clips
        "processing": 70,               # Creating video clips with FFmpeg
        
        # Step 5: Clips download/upload (final stages)
        "uploading_clips": 90,          # Uploading clips to S3
        
        # Step 6: Finished
        "completed": 100,               # All done!
        
        # Error states
        "failed": 0,
        "error": 0,
        "timeout": 0,
        "stuck": 0
    }
    return progress_map.get(status, 0)


def get_message_from_status(status: str, error_message: str = None) -> str:
    """Map video status to user-friendly message - aligned with actual pipeline"""
    if status == "failed" and error_message:
        return f"Error: {error_message}"
    if status == "error" and error_message:
        return f"Error: {error_message}"
    
    message_map = {
        # Step 1: Downloading YouTube video
        "queued": "Queued for processing...",
        "downloading": "Downloading YouTube video...",
        "downloaded": "Download complete",
        "uploading": "Uploading video to storage...",
        
        # Step 2: Transcribing
        "processing_started": "Starting video processing...",
        "transcribing": "Transcribing audio...",
        
        # Step 3: Finding interesting content
        "analyzing": "Finding interesting content...",
        
        # Step 4: Generating clips
        "processing": "Generating clips...",
        
        # Step 5: Clips download (upload to S3)
        "uploading_clips": "Uploading clips...",
        
        # Step 6: Finished
        "completed": "Finished!",
        
        # Error states
        "failed": "Processing failed",
        "error": "An error occurred",
        "timeout": "Processing timed out",
        "stuck": "Processing stuck"
    }
    return message_map.get(status, "Processing...")


@app.get("/clips/{video_id}")
async def get_clips(video_id: str):
    """Get clips for a video from the database"""
    try:
        # First try the new user video service structure
        from services.user_video_service import get_user_video_by_video_id
        video_info = await get_user_video_by_video_id(video_id)
        
        if video_info:
            # Get clips from the video document
            clips_data = video_info.get('clips', [])
            clips = []
            
            for clip_doc in clips_data:
                clip = {
                    "id": str(clip_doc.get("id")),
                    "title": clip_doc.get("title"),
                    "start_time": clip_doc.get("start_time", 0),
                    "end_time": clip_doc.get("end_time", 0),
                    "duration": clip_doc.get("duration", 0),
                    "s3_key": clip_doc.get("s3_key"),
                    "s3_url": clip_doc.get("s3_url"),
                    "thumbnail_url": clip_doc.get("thumbnail_url"),
                    "transcription": clip_doc.get("transcription", ""),
                    "created_at": clip_doc.get("created_at").isoformat() if clip_doc.get("created_at") else None
                }
                clips.append(clip)
            
            response = {"clips": clips}
            
            # Validate response URLs
            assert_response_urls(response, "clips endpoint response")
            
            logging.info(f"Retrieved {len(clips)} clips for video {video_id}")
            return response
        
        # Fallback to old structure if video not found in new structure
        db = get_database()
        clips_collection = db.clips
        
        # Fetch clips for the video from database
        clips_cursor = clips_collection.find({"video_id": video_id})
        clips = []
        
        for clip_doc in clips_cursor:
            clip = {
                "id": str(clip_doc.get("_id")),
                "title": clip_doc.get("title"),
                "start_time": clip_doc.get("start_time", 0),
                "end_time": clip_doc.get("end_time", 0),
                "duration": clip_doc.get("duration", 0),
                "s3_key": clip_doc.get("s3_key"),
                "s3_url": clip_doc.get("s3_url"),
                "thumbnail_url": clip_doc.get("thumbnail_url"),
                "transcription": clip_doc.get("transcription", ""),
                "created_at": clip_doc.get("created_at").isoformat() if clip_doc.get("created_at") else None
            }
            clips.append(clip)
        
        response = {"clips": clips}
        
        # Validate response URLs
        assert_response_urls(response, "clips endpoint response")
        
        logging.info(f"Retrieved {len(clips)} clips for video {video_id}")
        return response
        
    except Exception as e:
        logging.error(f"Error fetching clips for video {video_id}: {e}")
        # Return empty list on error
        response = {"clips": []}
        return response


@app.post("/process-video/{video_id}")
async def process_video_endpoint(video_id: str, background_tasks: BackgroundTasks):
    """Start processing a previously uploaded video to generate clips"""
    # Find the original video
    video_dir = upload_dir / video_id
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Find the first video file in the directory
    video_files = list(video_dir.glob("*.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video file not found")

    video_path = str(video_files[0])

    print("video path: ", video_path)

    # Create a task for processing
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting video processing",
        "video_id": video_id,
        "file_path": video_path,
    }

    # Start background processing
    background_tasks.add_task(
        process_podcast, task_id=task_id, video_id=video_id, file_path=video_path, user_id=None
    )

    return {"task_id": task_id, "video_id": video_id, "status": "processing"}


@app.post("/generate-clip")
async def generate_clip(background_tasks: BackgroundTasks, request: ClipRequest):
    # Find the original video
    video_dir = upload_dir / request.video_id
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Find the first video file in the directory
    video_files = list(video_dir.glob("*.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video file not found")

    video_path = str(video_files[0])

    # Create a task for clip generation
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting clip generation",
        "video_id": request.video_id,
    }

    # Start background processing
    background_tasks.add_task(
        generate_clip_task,
        task_id=task_id,
        video_id=request.video_id,
        video_path=video_path,
        start_time=request.start_time,
        end_time=request.end_time,
        title=request.title,
    )

    return {"task_id": task_id}


@app.post("/s3-upload-url")
async def get_s3_upload_url(
    request: S3UploadRequest,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Generate a presigned URL for direct S3 upload from the frontend"""
    if not s3_client.available:
        raise HTTPException(status_code=501, detail="S3 direct upload not configured")

    # Generate a unique ID for this video
    video_id = str(uuid.uuid4())

    # Generate a presigned POST URL
    presigned_data = s3_client.generate_presigned_post(
        video_id=video_id, filename=request.filename, content_type=request.content_type
    )

    if not presigned_data:
        raise HTTPException(status_code=500, detail="Failed to generate upload URL")

    # Create video entry in user's videos array using the new service
    try:
        video_data = {
            "id": video_id,
            "title": request.filename,
            "description": f"Video uploaded via S3",
            "filename": request.filename,
            "s3_key": presigned_data["s3_key"],
            "s3_url": s3_client.get_object_url(presigned_data["s3_key"]),
            "status": VideoStatus.UPLOADING,
            "video_type": VideoType.UPLOAD,
            "content_type": request.content_type
        }
        
        # Save to user's videos array
        await create_user_video(current_user.id, video_data)
        logging.info(f"Created video entry for user {current_user.id} with video_id: {video_id}")
        
    except Exception as e:
        logging.error(f"Error creating video entry: {str(e)}")
        # Don't fail the upload URL generation if database save fails

    return {
        "upload_url": presigned_data["url"],
        "upload_fields": presigned_data["fields"],
        "video_id": video_id,
        "s3_key": presigned_data["s3_key"],
    }


@app.post("/register-s3-upload")
async def register_uploaded_file(
    current_user: Annotated[User, Depends(get_current_user)],
    video_id: str = Form(...), 
    s3_key: str = Form(...), 
    filename: str = Form(...)
):
    """Register a file that was directly uploaded to S3"""
    if not s3_client.available:
        raise HTTPException(status_code=501, detail="S3 direct upload not configured")

    # Create a record of this upload
    logging.info(
        f"Registering S3 upload: video_id={video_id}, s3_key={s3_key}, filename={filename}, user_id={current_user.id}"
    )

    # Update video information in user's videos array using the new service
    try:
        # Update the video with the final S3 URL and status
        video_update_data = {
            "s3_url": s3_client.get_object_url(s3_key),
            "status": VideoStatus.COMPLETED,
            "updated_at": datetime.datetime.now()
        }
        
        await update_video_s3_url(current_user.id, video_id, video_update_data)
        logging.info(f"Updated S3 upload video {video_id} for user {current_user.id}")
        
    except Exception as e:
        logging.error(f"Error updating S3 upload video in database: {e}")
        # Don't fail the registration if database save fails

    return {
        "video_id": video_id,
        "s3_key": s3_key,
        "filename": filename,
        "status": "uploaded",
    }


@app.post("/process-s3-video/{video_id}")
async def process_s3_video(
    video_id: str, 
    s3_key: str, 
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Start processing a video that was uploaded to S3"""
    if not s3_client.available:
        raise HTTPException(status_code=501, detail="S3 direct upload not configured")

    try:
        # Create a local directory for this video
        video_dir = upload_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Get the filename from the S3 key
        filename = os.path.basename(s3_key)
        local_path = str(video_dir / filename)

        # Create a task for processing immediately
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Downloading file from S3",
            "video_id": video_id,
            "file_path": local_path,
            "s3_key": s3_key,
        }

        # Start background processing that will first download the file
        background_tasks.add_task(
            process_s3_podcast,
            task_id=task_id,
            video_id=video_id,
            s3_key=s3_key,
            local_path=local_path,
            user_id=current_user.id,
        )

        # Return immediately with task ID
        return {"task_id": task_id, "video_id": video_id, "status": "downloading"}

    except Exception as e:
        logging.error(f"Error processing S3 video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/youtube-download")
async def download_from_youtube(
    background_tasks: BackgroundTasks, 
    request: YouTubeRequest,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Download a video from YouTube locally and process it (no S3 upload of original video)"""
    try:
        # Generate a unique ID for the video
        video_id = str(uuid.uuid4())
        now = datetime.datetime.now()
        # Immediately create the video record in the DB
        video_data = {
            "id": video_id,
            "title": f"YouTube Video {video_id}",
            "description": f"Video submitted for processing",
            "status": "downloading",
            "video_type": "youtube",
            "user_id": current_user.id,
            "created_at": now,
            "updated_at": now,
        }
        await create_user_video(current_user.id, video_data)
        # Create a task ID
        task_id = str(uuid.uuid4())
        # Initialize task status
        tasks[task_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Starting YouTube download (local)...",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "video_id": video_id,
            "user_id": current_user.id,
        }
        # Start download in background (local processing, no S3 for original video)
        background_tasks.add_task(
            process_youtube_download_local,
            task_id=task_id,
            video_id=video_id,
            url=str(request.url),
            auto_process=request.auto_process,
            user_id=current_user.id,
        )
        return {"video_id": video_id, "task_id": task_id, "status": "downloading"}
    except Exception as e:
        logging.error(f"Error in YouTube download: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error downloading YouTube video: {str(e)}"
        )


async def process_youtube_download_local(
    task_id: str, video_id: str, url: str, auto_process: bool = True, user_id: str = None
):
    """Process a YouTube download locally (no S3 upload of original video)"""
    try:
        # Import user video service
        from services.user_video_service import update_user_video
        
        # Update video status in MongoDB (primary source of truth)
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "downloading",
                "updated_at": utc_now()
            })
        
        # Also update in-memory tasks for backwards compatibility
        tasks[task_id] = {
            "status": "downloading",
            "progress": 10,
            "message": "Downloading video from YouTube locally...",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "video_id": video_id,
        }
        logging.info(f"Task {task_id}: Starting local YouTube download for URL: {url}")

        # Create video directory
        video_dir = upload_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Download the video locally using Sieve or fallback downloader
        file_path, title, video_info = None, None, None
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Task {task_id}: Attempt {attempt} to download with Sieve service")
                file_path, title, video_info = await download_youtube_video_sieve(url, video_dir)
                if file_path and title:
                    break
            except Exception as sieve_error:
                logging.warning(f"Task {task_id}: Sieve download failed (attempt {attempt}): {str(sieve_error)}")
                if attempt == max_retries:
                    logging.error(f"Task {task_id}: All {max_retries} Sieve attempts failed.")
                    break
                await asyncio.sleep(5)
        
        # Fallback to direct downloader if Sieve failed
        if not file_path or not title:
            for attempt in range(1, max_retries + 1):
                try:
                    logging.info(f"Task {task_id}: Attempt {attempt} to download with direct downloader")
                    file_path, title, video_info = await download_youtube_video(url, video_dir)
                    if file_path and title:
                        break
                except Exception as fallback_error:
                    logging.warning(f"Task {task_id}: Direct download failed (attempt {attempt}): {str(fallback_error)}")
                    if attempt == max_retries:
                        logging.error(f"Task {task_id}: All {max_retries} fallback attempts failed.")
                        break
                    await asyncio.sleep(5)

        if not file_path or not title:
            raise Exception(f"Failed to download YouTube video after {max_retries} attempts")

        logging.info(f"Task {task_id}: Download completed successfully to local file: {file_path}")
        
        # Update video in database
        try:
            video_data = {
                "title": title or f"YouTube Video",
                "description": f"YouTube video downloaded from {url}",
                "filename": Path(file_path).name,
                "source_url": url,
                "status": "downloaded",
                "video_type": "youtube",
                "duration": video_info.get("length_seconds") if video_info else None,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "content_type": "video/mp4"
            }
            from services.user_video_service import update_user_video
            await update_user_video(user_id, video_id, video_data)
            logging.info(f"Updated YouTube video {video_id} in database.")
        except Exception as e:
            logging.error(f"Error updating YouTube video in database: {e}", exc_info=True)
        
        # Update task
        tasks[task_id] = {
            "status": "downloaded",
            "progress": 100,
            "message": "YouTube video downloaded successfully (local)",
            "video_info": {
                "video_id": video_id,
                "filename": Path(file_path).name,
                "title": title,
                "thumbnail": video_info.get("thumbnail_url") if video_info else None,
                "duration": video_info.get("length_seconds") if video_info else None,
            },
            "updated_at": datetime.datetime.now().isoformat(),
            "video_id": video_id,
        }

        # Start processing if requested
        if auto_process:
            try:
                process_task_id = str(uuid.uuid4())
                logging.info(f"Task {task_id}: Created process task ID: {process_task_id}")
                
                tasks[process_task_id] = {
                    "status": "queued",
                    "progress": 0,
                    "message": "Queued for processing",
                    "created_at": datetime.datetime.now().isoformat(),
                    "updated_at": datetime.datetime.now().isoformat(),
                    "video_id": video_id,
                    "original_task_id": task_id,
                }
                
                tasks[task_id]["process_task_id"] = process_task_id
                tasks[task_id]["status"] = "processing_started"
                tasks[task_id]["message"] = "Processing has started"
                tasks[task_id]["updated_at"] = datetime.datetime.now().isoformat()
                
                # Process directly from local file (no S3 download needed)
                await process_podcast(
                    process_task_id, video_id, file_path, original_task_id=task_id, user_id=user_id
                )
                
            except Exception as process_error:
                logging.error(f"Error starting processing for task {task_id}: {str(process_error)}", exc_info=True)
                tasks[task_id].update({
                    "status": "error",
                    "message": f"Error starting processing: {str(process_error)}",
                    "updated_at": datetime.datetime.now().isoformat(),
                })
                
    except Exception as e:
        logging.error(f"Error in YouTube download process: {str(e)}", exc_info=True)
        tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error processing YouTube download: {str(e)}",
            "updated_at": datetime.datetime.now().isoformat(),
        }


# DEPRECATED: Old S3-based YouTube download (kept for reference, not used)
async def process_youtube_download_s3(
    task_id: str, video_id: str, url: str, auto_process: bool = True, user_id: str = None
):
    """DEPRECATED: Process a YouTube download directly to S3 (no longer used - we process locally)"""
    try:
        # Update task status
        tasks[task_id] = {
            "status": "downloading",
            "progress": 10,
            "message": "Downloading video from YouTube to S3...",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "video_id": video_id,
        }
        logging.info(f"Task {task_id}: Starting YouTube download to S3 for URL: {url}")

        # Download directly to S3
        success, s3_key, video_info = await s3_client.download_youtube_to_s3(url, video_id)
        
        if not success or not s3_key:
            logging.error(f"Task {task_id}: Failed to download YouTube video to S3")
            tasks[task_id] = {
                "status": "error",
                "progress": 0,
                "message": "Failed to download YouTube video to S3",
                "updated_at": datetime.datetime.now().isoformat(),
            }
            return

        # Log the successful download
        logging.info(f"Task {task_id}: Download completed successfully to S3: {s3_key}")
        
        # Save video information to database using the new user video service
        try:
            # Extract video ID from URL for title
            video_id_from_url = url.split('/')[-1] if url else "unknown"
            
            logging.info(f"Attempting to update video in database with user_id: {user_id}")
            
            # Update video data for user's videos array
            video_data = {
                "title": video_info.get("title", f"YouTube Video {video_id_from_url}"),
                "description": f"YouTube video downloaded from {url}",
                "filename": video_info.get("filename", f"youtube_{video_id_from_url}.mp4"),
                "s3_key": s3_key,
                "s3_url": s3_client.get_object_url(s3_key),
                "source_url": url,
                "status": "processing",  # or "completed" if done
                "video_type": "youtube",
                "duration": video_info.get("duration"),
                "file_size": video_info.get("file_size"),
                "content_type": "video/mp4"
            }
            from services.user_video_service import update_user_video
            await update_user_video(user_id, video_id, video_data)
            logging.info(f"Updated YouTube video {video_id} in database using new service.")
        except Exception as e:
            logging.error(f"Error updating YouTube video in database: {e}", exc_info=True)
            # Don't fail the download if database save fails
        
        # Update task with S3 information
        tasks[task_id] = {
            "status": "downloaded",
            "progress": 100,
            "message": "YouTube video downloaded successfully to S3",
            "video_info": {
                "video_id": video_id,
                "s3_key": s3_key,
                "s3_url": video_info.get("s3_url"),
                "filename": video_info.get("filename"),
                "title": video_info.get("title"),
                "thumbnail": video_info.get("thumbnail_url"),
                "duration": video_info.get("length_seconds"),
            },
            "updated_at": datetime.datetime.now().isoformat(),
            "video_id": video_id,
        }

        # Add a small delay before starting processing
        logging.info(f"Task {task_id}: Waiting 3 seconds before starting processing")
        await asyncio.sleep(3)

        # Start processing if requested
        if auto_process:
            try:
                # Create a new task ID for processing
                process_task_id = str(uuid.uuid4())
                logging.info(f"Task {task_id}: Created process task ID: {process_task_id}")
                
                # Initialize processing task
                tasks[process_task_id] = {
                    "status": "queued",
                    "progress": 0,
                    "message": "Queued for processing",
                    "created_at": datetime.datetime.now().isoformat(),
                    "updated_at": datetime.datetime.now().isoformat(),
                    "video_id": video_id,
                    "original_task_id": task_id,
                }
                
                # Update the original task with the processing task ID
                tasks[task_id]["process_task_id"] = process_task_id
                tasks[task_id]["status"] = "processing_started"
                tasks[task_id]["message"] = "Processing has started"
                tasks[task_id]["updated_at"] = datetime.datetime.now().isoformat()
                
                # Start processing with S3 key
                await process_s3_podcast(
                    process_task_id, video_id, s3_key, None, original_task_id=task_id, user_id=user_id
                )
                
            except Exception as process_error:
                logging.error(f"Error starting processing for task {task_id}: {str(process_error)}", exc_info=True)
                tasks[task_id].update({
                    "status": "error",
                    "message": f"Error starting processing: {str(process_error)}",
                    "updated_at": datetime.datetime.now().isoformat(),
                })
                
    except Exception as e:
        logging.error(f"Error in YouTube download process: {str(e)}", exc_info=True)
        tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error processing YouTube download: {str(e)}",
            "updated_at": datetime.datetime.now().isoformat(),
        }

# New background task for S3 processing
async def process_s3_podcast(task_id: str, video_id: str, s3_key: str, local_path: str = None, original_task_id: Optional[str] = None, user_id: str = None):
    try:
        # Create temporary file for processing if not provided
        if not local_path:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                local_path = temp_file.name

        # Download the file from S3
        logging.info(f"Downloading file from S3: {s3_key} to {local_path}")
        success = s3_client.download_from_s3(s3_key, local_path)

        if not success:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = "Failed to download file from S3"
            logging.error(f"Failed to download file from S3: {s3_key}")
            return

        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "File downloaded, starting processing"
        tasks[task_id]["progress"] = 5

        # Now continue with regular processing
        await process_podcast(task_id, video_id, local_path, original_task_id, user_id)

    except Exception as e:
        logging.error(f"Error in S3 processing: {str(e)}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = f"Error: {str(e)}"
    finally:
        # Clean up temporary file if we created it
        if local_path and not local_path.startswith(str(upload_dir)):
            try:
                os.unlink(local_path)
            except:
                pass


# Background tasks
async def process_podcast(
    task_id: str, video_id: str, file_path: str, original_task_id: Optional[str] = None, user_id: str = None
):
    try:
        global whisper_model
        logging.info(
            f"Starting podcast processing for task {task_id}, video {video_id}"
        )

        # Update task status
        tasks[task_id] = {
            "status": "transcribing",
            "progress": 30,
            "message": "Transcribing audio",
            "video_id": video_id,
            "user_id": user_id,  # Store user_id in task
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
        }

        # Also update the original task if it exists
        if original_task_id and original_task_id in tasks:
            tasks[original_task_id].update(
                {
                    "status": "transcribing",
                    "progress": 30,
                    "message": "Transcribing audio",
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )

        logging.info(f"Starting transcription for {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Video file not found at path: {file_path}")
            raise FileNotFoundError(f"Video file not found at path: {file_path}")

        # Transcribe the audio using the pre-loaded model
        try:
            # Make transcription non-blocking by running in thread pool
            import asyncio
            import functools
            from concurrent.futures import ThreadPoolExecutor

            # Use limited thread pool to prevent resource exhaustion
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="whisper-worker"
            )
            loop = asyncio.get_event_loop()

            # Run transcription in separate thread to keep API responsive
            transcript = await loop.run_in_executor(
                executor, functools.partial(
                    transcribe_audio_sync, 
                    file_path, 
                    "tiny",
                    enable_diarization=True  # Enable speaker diarization
                )
            )
            if not transcript:
                raise ValueError("Transcription returned None")

            # Print the transcript output to the console for debugging
            print("\n--- TRANSCRIPTION OUTPUT ---\n", transcript, "\n--- END TRANSCRIPTION OUTPUT ---\n")

            # Handle empty segments gracefully - this is valid for silent/corrupted videos
            segments_count = len(transcript.get("segments", []))
            logging.info(f"Transcription completed with {segments_count} segments")

            if segments_count == 0:
                logging.warning("Video appears to be silent or have no speech content")
                # Continue processing with empty transcript - don't fail
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")

            # Try one more time with CPU-only mode
            logging.info("Retrying transcription with CPU-only mode")
            try:
                # Force CPU transcription by temporarily setting CUDA_VISIBLE_DEVICES
                original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

                # Reload the model on CPU
                from services.transcription import load_model

                cpu_model = load_model(model_size="tiny")

                # Transcribe with CPU

                audio_path = file_path
                if not file_path.lower().endswith((".mp3", ".wav", ".flac", ".aac")):
                    from services.transcription import extract_audio

                    audio_path = extract_audio(file_path)

                try:
                    from services.transcription import safe_whisper_transcribe

                    transcript = safe_whisper_transcribe(
                        cpu_model, audio_path, fp16=False
                    )
                except Exception as transcribe_error:
                    error_str = str(transcribe_error).lower()
                    # Check for empty audio related errors
                    if any(
                        pattern in error_str
                        for pattern in [
                            "reshape tensor",
                            "0 elements",
                            "linear(",
                            "unknown parameter type",
                            "dimension size -1",
                            "ambiguous",
                            "in_features",
                            "out_features",
                            "cannot reshape",
                            "unspecified dimension",
                            "failed to load audio",
                            "ffmpeg version",
                            "could not open",
                            "invalid data found",
                        ]
                    ):
                        logging.info(
                            "CPU transcription detected empty/silent audio - using empty transcript"
                        )
                        transcript = {"text": "", "segments": [], "language": "en"}
                    else:
                        raise transcribe_error

                # Restore original CUDA settings
                if original_cuda_devices is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

                # Don't fail if we have empty segments - that's OK for silent videos
                if not transcript:
                    raise ValueError("CPU transcription returned None")

                logging.info(
                    f"CPU transcription completed with {len(transcript.get('segments', []))} segments"
                )
            except Exception as cpu_error:
                logging.error(f"CPU transcription also failed: {str(cpu_error)}")

                # If CPU also fails with empty audio errors, create empty transcript
                error_str = str(cpu_error).lower()
                if any(
                    pattern in error_str
                    for pattern in [
                        "reshape tensor",
                        "0 elements",
                        "linear(",
                        "unknown parameter type",
                        "dimension size -1",
                        "ambiguous",
                        "in_features",
                        "out_features",
                        "cannot reshape",
                        "unspecified dimension",
                        "failed to load audio",
                        "ffmpeg version",
                        "could not open",
                        "invalid data found",
                    ]
                ):
                    logging.info(
                        "Both transcription attempts failed with empty audio - proceeding with empty transcript"
                    )
                    transcript = {"text": "", "segments": [], "language": "en"}
                else:
                    raise ValueError(
                        f"Transcription failed after multiple attempts: {str(e)}"
                    )

        # Update task status
        tasks[task_id].update(
            {
                "status": "analyzing",
                "progress": 50,
                "message": "Analyzing content",
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )

        # Also update the original task if it exists
        if original_task_id and original_task_id in tasks:
            tasks[original_task_id].update(
                {
                    "status": "analyzing",
                    "progress": 50,
                    "message": "Analyzing content",
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )

        logging.info("Starting content analysis to find interesting segments")

        # Analyze the content to find interesting segments
        try:
            segments = await analyze_content(transcript)
            if not segments:
                logging.warning(
                    "Content analysis found no interesting segments, using default segments"
                )
                # Create a default segment if none found (1 minute from the beginning)
                segments = [
                    {
                        "start_time": 0,
                        "end_time": 60,
                        "title": "Automatic Clip",
                        "description": "Automatically generated clip",
                    }
                ]
            logging.info(
                f"Content analysis completed, found {len(segments)} interesting segments"
            )
        except Exception as e:
            logging.error(f"Content analysis failed: {str(e)}")
            raise

        # Update task status
        tasks[task_id].update(
            {
                "status": "processing",
                "progress": 70,
                "message": "Processing video clips",
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )

        # Also update the original task if it exists
        if original_task_id and original_task_id in tasks:
            tasks[original_task_id].update(
                {
                    "status": "processing",
                    "progress": 70,
                    "message": "Processing video clips",
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )

        logging.info("Starting video processing to create clips")

        # Create output directory for this video
        output_dir.mkdir(exist_ok=True)  # Ensure parent directory exists
        clips_dir = output_dir / video_id
        clips_dir.mkdir(exist_ok=True)

        # Process video clips using the improved pipeline
        try:
            logging.info(f"Starting video processing with {len(segments)} segments")
            clips = await process_video(file_path, transcript, segments)
            logging.info(f"Video processing completed, created {len(clips)} clips")
            logging.info(f"Clips data: {clips}")

            # If no clips were created, create a simple clip
            if not clips:
                logging.warning(
                    "No clips were created by the video processing pipeline. Creating a simple clip."
                )

                # Create a simple clip using basic FFmpeg command
                simple_clip_id = "simple_clip"
                simple_clip_path = str(clips_dir / f"{simple_clip_id}.mp4")

                # Use a simple command without face tracking
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            "0",
                            "-i",
                            file_path,
                            "-t",
                            "60",
                            "-c:v",
                            "libx264",
                            "-c:a",
                            "aac",
                            simple_clip_path,
                        ],
                        check=True,
                    )

                    # Add a simple clip to the list
                    clips = [
                        {
                            "id": simple_clip_id,
                            "start_time": 0,
                            "end_time": 60,
                            "title": "Simple Clip",
                            "description": "Automatically generated simple clip",
                            "path": simple_clip_path,
                            "url": f"/outputs/{video_id}/{os.path.basename(simple_clip_path)}",
                        }
                    ]
                    logging.info("Created a simple clip as fallback")
                except Exception as simple_clip_error:
                    logging.error(
                        f"Failed to create simple clip: {str(simple_clip_error)}"
                    )
                    # Continue with empty clips list if this fails too
        except Exception as e:
            logging.error(f"Video processing failed: {str(e)}")
            # Continue with empty clips list
            clips = []

        # ====================================================================
        # PARALLEL CLIP PROCESSING - Process multiple clips at same time
        # ====================================================================
        
        if clips:
            logging.info(f"[PARALLEL] Starting parallel processing of {len(clips)} clips")
            
            import time
            parallel_start_time = time.time()
            
            # Prepare for parallel processing
            MAX_CONCURRENT_CLIPS = 3  # Adjust based on system resources
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)
            
            async def bounded_process_clip(clip, idx):
                """Process clip with concurrency limit"""
                async with semaphore:
                    return await process_single_clip_async(
                        clip, idx, file_path, video_id, output_dir, s3_client
                    )
            
            # Update progress - starting parallel processing
            tasks[task_id].update(
                {
                    "progress": "60",
                    "message": f"Processing {len(clips)} clips in parallel...",
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )
            
            if original_task_id and original_task_id in tasks:
                tasks[original_task_id].update(
                    {
                        "progress": "60",
                        "message": f"Processing {len(clips)} clips in parallel...",
                        "updated_at": datetime.datetime.now().isoformat(),
                    }
                )
            
            # Create all clip processing tasks
            clip_tasks = [
                bounded_process_clip(clip, i)
                for i, clip in enumerate(clips)
            ]
            
            # Run all tasks in parallel
            logging.info(f"[PARALLEL] Running {len(clip_tasks)} clips in parallel (max {MAX_CONCURRENT_CLIPS} concurrent)")
            results = await asyncio.gather(*clip_tasks, return_exceptions=True)
            
            # Process results
            logging.info(f"[PARALLEL] All clips processed, updating results...")
            for result in results:
                if result and isinstance(result, tuple):
                    index, data = result[0], result[1]
                    if data:
                        clips[index]['s3_key'] = data['clip_s3_key']
                        clips[index]['s3_url'] = s3_client.get_object_url(data['clip_s3_key']) if data['clip_s3_key'] else None
                        clips[index]['thumbnail_s3_key'] = data['thumbnail_s3_key']
                        clips[index]['thumbnail_url'] = s3_client.get_object_url(data['thumbnail_s3_key']) if data['thumbnail_s3_key'] else "/static/default_thumbnail.jpg"
                        logging.info(f"[PARALLEL] ✅ Clip {index} complete - S3 URL: {clips[index].get('s3_url')}")
                    else:
                        logging.warning(f"[PARALLEL] ❌ Clip {index} returned None results")
                        clips[index]['thumbnail_url'] = "/static/default_thumbnail.jpg"
                elif isinstance(result, Exception):
                    logging.error(f"[PARALLEL] ❌ Clip raised exception: {result}")
                    import traceback
                    logging.error(f"[PARALLEL] Traceback: {traceback.format_exc()}")
                else:
                    logging.warning(f"[PARALLEL] ❌ Clip returned invalid result type")
            
            parallel_elapsed = time.time() - parallel_start_time
            logging.info(f"[PARALLEL] ⏱️  Total time: {parallel_elapsed:.1f}s")
            logging.info(f"[PARALLEL] 📊 Average per clip: {parallel_elapsed/len(clips):.1f}s")
            if len(clips) > 1:
                sequential_estimate = parallel_elapsed * len(clips) / MAX_CONCURRENT_CLIPS
                speedup = sequential_estimate / parallel_elapsed if parallel_elapsed > 0 else 0
                logging.info(f"[PARALLEL] 📈 Estimated speedup vs sequential: ~{speedup:.1f}x")
            
            logging.info(f"[PARALLEL] ✅ Parallel processing complete for all {len(clips)} clips")
        else:
            logging.info("No clips to process")

        # Validate S3 URLs in clips
        logging.info(f"Starting S3 URL validation for {len(clips)} clips")
        logging.info(f"S3 client available: {s3_client.available}")
        if s3_client.available:
            logging.info(f"S3 bucket: {s3_client.bucket}")
        
        for i, clip in enumerate(clips):
            logging.info(f"Validating clip {i+1}/{len(clips)}: {clip.get('id')}")
            
            if "s3_url" in clip and clip["s3_url"]:
                logging.info(f"Validating S3 URL for clip {clip.get('id')}: {clip['s3_url']}")
                try:
                    assert_s3_url(clip["s3_url"], f"clip {clip.get('id')} s3_url")
                    logging.info(f"S3 URL validation passed for clip {clip.get('id')}")
                except Exception as e:
                    logging.error(f"S3 URL validation failed for clip {clip.get('id')}: {e}")
                    logging.error(f"Clip data: {clip}")
                    raise
            else:
                logging.warning(f"No s3_url found for clip {clip.get('id')}")
                
            if "thumbnail_url" in clip and clip["thumbnail_url"].startswith("http"):
                logging.info(f"Validating thumbnail URL for clip {clip.get('id')}: {clip['thumbnail_url']}")
                try:
                    assert_s3_url(clip["thumbnail_url"], f"clip {clip.get('id')} thumbnail_url")
                    logging.info(f"Thumbnail URL validation passed for clip {clip.get('id')}")
                except Exception as e:
                    logging.error(f"Thumbnail URL validation failed for clip {clip.get('id')}: {e}")
                    logging.error(f"Clip data: {clip}")
                    raise
            else:
                logging.info(f"No thumbnail URL to validate for clip {clip.get('id')}")
        
        logging.info("S3 URL validation completed successfully")

        # Save clips to database using the new user video service
        logging.info("Saving clips to database using new service...")
        logging.info(f"Total clips to save: {len(clips)}")
        
        try:
            # Get user_id from task
            task_data = tasks.get(task_id, {})
            user_id = task_data.get("user_id")
            
            if not user_id:
                logging.warning(f"No user_id found in task for video {video_id}")
                # Try to get from video record in old format
                db = get_database()
                videos_collection = db.videos
                video_record = videos_collection.find_one({"video_id": video_id})
                if video_record:
                    user_id = video_record.get("user_id")
                    logging.info(f"Found user_id from old video record: {user_id}")
                
                # If still not found, try to find video in new structure
                if not user_id:
                    from services.user_video_service import get_user_video_by_video_id
                    try:
                        video_info = await get_user_video_by_video_id(video_id)
                        if video_info:
                            user_id = video_info.get("user_id")
                            logging.info(f"Found user_id from new video structure: {user_id}")
                    except Exception as e:
                        logging.warning(f"Could not find video in new structure: {e}")
            
            if not user_id:
                logging.error(f"Cannot save clips to database - no user_id available for video {video_id}")
                logging.info("Skipping clip saving - continuing with processing")
                return
            
            # Ensure video exists in database before processing clips
            from services.user_video_service import get_user_video
            existing_video = await get_user_video(user_id, video_id)
            if not existing_video:
                logging.error(f"Video {video_id} not found in database. Skipping clip saving and processing for this video.")
                return
            
            # Save each clip to the user's video using the new service
            clips_saved = 0
            for i, clip in enumerate(clips):
                logging.info(f"Processing clip {i+1}/{len(clips)}: {clip.get('id', 'unknown')}")
                logging.info(f"Clip data: {clip}")
                
                if clip.get("s3_url"):  # Only save clips that have S3 URLs
                    logging.info(f"Clip has S3 URL: {clip.get('s3_url')}")
                    clip_data = {
                        "title": clip.get("title", f"Clip {clip.get('id')}"),
                        "start_time": clip.get("start_time", 0),
                        "end_time": clip.get("end_time", 0),
                        "s3_key": clip.get("s3_key"),
                        "s3_url": clip.get("s3_url"),
                        "thumbnail_url": clip.get("thumbnail_url"),
                        "transcription": clip.get("transcription", ""),
                        "summary": clip.get("summary", ""),
                        "tags": clip.get("tags", []),
                        "metadata": clip.get("metadata", {})
                    }
                    
                    logging.info(f"Prepared clip data: {clip_data}")
                    
                    try:
                        # Add clip to user's video using the new service
                        clip_id = await add_clip_to_video(user_id, video_id, clip_data)
                        if clip_id:
                            logging.info(f"✅ Saved clip {clip.get('id')} to database with ID: {clip_id}")
                            clips_saved += 1
                        else:
                            logging.error(f"❌ Failed to save clip {clip.get('id')} to database")
                    except Exception as clip_save_error:
                        logging.error(f"❌ Error saving clip {clip.get('id')} to database: {clip_save_error}")
                        import traceback
                        logging.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logging.warning(f"⚠️ Skipping clip {clip.get('id')} - no S3 URL")
                    logging.warning(f"Clip data: {clip}")
            
            logging.info(f"✅ Successfully saved {clips_saved}/{len(clips)} clips to database using new service")
            
        except Exception as e:
            logging.error(f"❌ Error saving clips to database: {e}", exc_info=True)
            # Don't fail the entire process if database save fails
        
        # Save metadata to S3 (optional, since clips are now in S3)
        metadata = {"video_id": video_id, "clips": clips}
        logging.info(f"Processing completed with {len(clips)} clips")

        # Update task status
        tasks[task_id].update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Processing completed",
                "clips": clips,
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )
        logging.info(f"Task {task_id} completed successfully with {len(clips)} clips")

        # Update the video record in the database as completed
        try:
            from services.user_video_service import update_user_video
            processed_at = datetime.datetime.now()
            # Find a representative thumbnail for the video (first clip with a thumbnail)
            video_thumbnail_url = None
            for clip in clips:
                if clip.get("thumbnail_url") and clip["thumbnail_url"].startswith("http"):
                    video_thumbnail_url = clip["thumbnail_url"]
                    break
            video_update_data = {
                "status": "completed",
                "processed_at": processed_at,
                "thumbnail_url": video_thumbnail_url,
                # Optionally add more fields here (e.g., s3_url, etc.)
            }
            if user_id:
                await update_user_video(user_id, video_id, video_update_data)
                logging.info(f"Updated video {video_id} in DB as completed.")
            else:
                logging.warning(f"No user_id available to update video {video_id} in DB.")
        except Exception as e:
            logging.error(f"Error updating video {video_id} in DB as completed: {e}")
        
        # Save metadata to S3 (optional, since clips are now in S3)
        metadata = {"video_id": video_id, "clips": clips}
        logging.info(f"Processing completed with {len(clips)} clips")

        # Update task status
        tasks[task_id].update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Processing completed",
                "clips": clips,
                "updated_at": datetime.datetime.now().isoformat(),
            }
        )
        logging.info(f"Task {task_id} completed successfully with {len(clips)} clips")
        
        # Create clips_info.json and upload to S3
        try:
            clips_info = {
                "video_id": video_id,
                "total_clips": len(clips),
                "clips": []
            }
            
            for clip in clips:
                clip_info = {
                    "id": clip.get("id"),
                    "title": clip.get("title", ""),
                    "description": clip.get("description", ""),
                    "start_time": clip.get("start_time", 0),
                    "end_time": clip.get("end_time", 0),
                    "duration": clip.get("duration", 0),
                    "s3_url": clip.get("s3_url"),
                    "thumbnail_url": clip.get("thumbnail_url"),
                    "transcription": clip.get("transcription", ""),
                    "created_at": datetime.datetime.now().isoformat()
                }
                clips_info["clips"].append(clip_info)
            
            # Save clips_info.json locally
            clips_info_path = clips_dir / "clips_info.json"
            with open(clips_info_path, 'w') as f:
                json.dump(clips_info, f, indent=2)
            
            # Upload clips_info.json to S3
            success, clips_info_s3_key = s3_client.upload_file_to_s3(
                str(clips_info_path), video_id, "clips_info.json", "application/json"
            )
            
            if success:
                clips_info_s3_url = s3_client.get_object_url(clips_info_s3_key)
                logging.info(f"clips_info.json uploaded to S3: {clips_info_s3_url}")
                
                # Clean up local clips_info.json
                try:
                    os.unlink(clips_info_path)
                except:
                    pass
            else:
                logging.error("Failed to upload clips_info.json to S3")
                
        except Exception as clips_info_error:
            logging.error(f"Error creating/uploading clips_info.json: {clips_info_error}")

        # Clean up local video_id directory after successful S3 upload
        try:
            clips_dir = output_dir / video_id
            if os.path.exists(clips_dir):
                import shutil
                shutil.rmtree(clips_dir)
                logging.info(f"Cleaned up local video directory: {clips_dir}")
        except Exception as cleanup_error:
            logging.warning(f"Failed to clean up local video directory {clips_dir}: {cleanup_error}")

        # If this was started from another task, update that task too
        if original_task_id and original_task_id in tasks:
            tasks[original_task_id].update(
                {
                    "status": "completed",
                    "progress": 100,
                    "message": "Processing completed",
                    "clips": clips,
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )
            logging.info(
                f"Original task {original_task_id} also updated to completed status"
            )

    except Exception as e:
        # Update task status on error
        tasks[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}",
            "updated_at": datetime.datetime.now().isoformat(),
        }
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        import traceback

        logging.error(traceback.format_exc())

        # If this was started from another task, update that task too
        if original_task_id and original_task_id in tasks:
            tasks[original_task_id].update(
                {
                    "status": "error",
                    "progress": 0,
                    "message": f"Error: {str(e)}",
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )


async def generate_clip_task(
    task_id: str,
    video_id: str,
    video_path: str,
    start_time: float,
    end_time: float,
    title: Optional[str],
):
    try:
        global whisper_model
        logging.info(
            f"Starting manual clip generation for task {task_id}, video {video_id}"
        )
        logging.info(
            f"Clip parameters: start_time={start_time}, end_time={end_time}, title={title or 'Manual Clip'}"
        )

        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Processing clip"
        tasks[task_id]["progress"] = 20

        # Generate a unique ID for this clip
        clip_id = f"manual_clip_{uuid.uuid4().hex[:8]}"

        # Create a manual segment
        manual_segment = {
            "start_time": start_time,
            "end_time": end_time,
            "title": title or "Manual Clip",
            "description": "Manually created clip",
        }

        # For manual clips, we don't need the full transcription pipeline
        # We can create a minimal transcript structure
        minimal_transcript = {"text": "", "segments": []}

        # Process the video clip using the improved pipeline
        logging.info(f"Processing manual clip using video_path={video_path}")
        clips = await process_video(video_path, minimal_transcript, [manual_segment])

        if not clips or len(clips) == 0:
            logging.error(
                "Failed to generate clip: process_video returned empty clips list"
            )
            raise Exception("Failed to generate clip")

        clip = clips[0]  # Get the first (and only) clip
        logging.info(f"Manual clip generated successfully: {clip.get('path')}")

        # Upload clip to S3
        clip_path = clip.get("path")
        if clip_path and os.path.exists(clip_path):
            # Upload clip to S3
            success, clip_s3_key = s3_client.upload_clip_to_s3(
                clip_path, video_id, clip_id, title
            )
            
            if success:
                clip_s3_url = s3_client.get_object_url(clip_s3_key)
                clip["s3_key"] = clip_s3_key
                clip["s3_url"] = clip_s3_url
                logging.info(f"Clip uploaded to S3: {clip_s3_url}")
                
                # Clean up local clip file after successful S3 upload
                try:
                    os.unlink(clip_path)
                    logging.info(f"Deleted local clip file: {clip_path}")
                except Exception as cleanup_error:
                    logging.warning(f"Failed to delete local clip file {clip_path}: {cleanup_error}")
            else:
                logging.error("Failed to upload clip to S3")
                raise Exception("Failed to upload clip to S3")

            # Generate thumbnail for this clip
            thumbnail_filename = f"{clip.get('id')}_thumbnail.jpg"
            
            # Create temporary thumbnail file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_thumb:
                thumbnail_path = temp_thumb.name

            try:
                # Generate thumbnail at middle of the clip but ensure it's within valid range
                clip_duration = end_time - start_time
                thumbnail_timestamp = (
                    min(clip_duration / 2, clip_duration - 0.5)
                    if clip_duration > 1.0
                    else 0
                )

                # Generate the thumbnail
                logging.info(
                    f"Generating thumbnail for clip {clip.get('id')} at path {thumbnail_path}"
                )
                generated_path = await generate_thumbnail(
                    clip_path, thumbnail_path, thumbnail_timestamp
                )

                # Verify the thumbnail exists and has content
                if (
                    generated_path
                    and os.path.exists(thumbnail_path)
                    and os.path.getsize(thumbnail_path) > 0
                ):
                    # Upload thumbnail to S3
                    thumb_success, thumb_s3_key = s3_client.upload_thumbnail_to_s3(
                        thumbnail_path, video_id, clip_id
                    )
                    
                    if thumb_success:
                        thumb_s3_url = s3_client.get_object_url(thumb_s3_key)
                        clip["thumbnail_s3_key"] = thumb_s3_key
                        clip["thumbnail_url"] = thumb_s3_url
                        logging.info(f"Thumbnail uploaded to S3: {thumb_s3_url}")
                    else:
                        logging.warning("Failed to upload thumbnail to S3, using default")
                        clip["thumbnail_url"] = "/static/default_thumbnail.jpg"
                else:
                    logging.warning(
                        f"Thumbnail generation returned success but file doesn't exist or is empty: {thumbnail_path}"
                    )
                    clip["thumbnail_url"] = "/static/default_thumbnail.jpg"
            except Exception as thumb_error:
                logging.error(f"Error generating thumbnail: {str(thumb_error)}")
                clip["thumbnail_url"] = "/static/default_thumbnail.jpg"
            finally:
                # Clean up temporary thumbnail file
                try:
                    os.unlink(thumbnail_path)
                except:
                    pass
        else:
            logging.warning(f"Clip file doesn't exist or is invalid: {clip_path}")
            clip["thumbnail_url"] = "/static/default_thumbnail.jpg"

        # Validate S3 URLs in clip data
        if "s3_url" in clip:
            assert_s3_url(clip["s3_url"], "clip s3_url")
        if "thumbnail_url" in clip and clip["thumbnail_url"].startswith("http"):
            assert_s3_url(clip["thumbnail_url"], "clip thumbnail_url")

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Clip generated successfully"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["clip"] = clip  # Use the clip data directly from process_video
        logging.info(
            f"Task {task_id} completed successfully with clip ID {clip.get('id')}"
        )

    except Exception as e:
        # Update task status on error
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = f"Error: {str(e)}"
        logging.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        import traceback

        logging.error(traceback.format_exc())





@app.get("/processing-status/{video_id}")
async def get_processing_status(video_id: str):
    """Get the processing status for a video"""
    try:
        # Find all tasks related to this video ID
        related_tasks = []
        for task_id, task_data in tasks.items():
            if task_data.get("video_id") == video_id:
                related_tasks.append(
                    {
                        "task_id": task_id,
                        "status": task_data.get("status", "unknown"),
                        "progress": task_data.get("progress", 0),
                        "message": task_data.get("message", ""),
                        "process_task_id": task_data.get("process_task_id", None),
                        "created_at": task_data.get("created_at", None),
                        "updated_at": task_data.get("updated_at", None),
                    }
                )

        if not related_tasks:
            raise HTTPException(
                status_code=404, detail="No tasks found for this video ID"
            )

        # Sort tasks by created_at (newest first)
        related_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Get the most recent task
        latest_task = related_tasks[0]

        # If the latest task has a process_task_id, get that task's status
        process_task_id = latest_task.get("process_task_id")
        if process_task_id and process_task_id in tasks:
            process_task = tasks[process_task_id]
            return {
                "video_id": video_id,
                "original_task_id": latest_task["task_id"],
                "process_task_id": process_task_id,
                "status": process_task.get("status", "unknown"),
                "progress": process_task.get("progress", 0),
                "message": process_task.get("message", ""),
                "has_processing_started": True,
            }

        # If no process_task_id or the process task doesn't exist, return the original task status
        return {
            "video_id": video_id,
            "original_task_id": latest_task["task_id"],
            "process_task_id": process_task_id,
            "status": latest_task["status"],
            "progress": latest_task["progress"],
            "message": latest_task["message"],
            "has_processing_started": latest_task["status"]
            in [
                "processing_started",
                "processing",
                "transcribing",
                "analyzing",
                "completed",
            ],
        }
    except Exception as e:
        logging.error(f"Error getting processing status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting processing status: {str(e)}"
        )


@app.get("/user/history/videos")
async def get_user_video_history(
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = 1,
    page_size: int = 10
):
    """Get user's video processing history"""
    try:
        # Use the new user video service
        result = await get_user_videos(current_user.id, page, page_size)
        
        return {
            "videos": [video.dict() for video in result["videos"]],
            "total_count": result["total_count"],
            "page": result["page"],
            "page_size": result["page_size"],
            "total_pages": result["total_pages"]
        }
    except Exception as e:
        logging.error(f"Error getting user video history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/user/history/clips")
async def get_user_clip_history(
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = 1,
    page_size: int = 10
):
    """Get user's clip history"""
    try:
        # Use the new user video service
        result = await get_user_clips(current_user.id, page, page_size)
        
        return {
            "clips": [clip.dict() for clip in result["clips"]],
            "total_count": result["total_count"],
            "page": result["page"],
            "page_size": result["page_size"],
            "total_pages": result["total_pages"]
        }
    except Exception as e:
        logging.error(f"Error getting user clip history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/user/history/summary")
async def get_user_history_summary(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Get user's processing history summary"""
    try:
        db = get_database()
        videos_collection = db.videos
        clips_collection = db.clips
        
        # Get counts
        total_videos = videos_collection.count_documents({"user_id": current_user.id})
        total_clips = clips_collection.count_documents({"user_id": current_user.id})
        
        # Get recent activity (last 7 days)
        from datetime import datetime, timedelta
        week_ago = datetime.now() - timedelta(days=7)
        
        recent_videos = videos_collection.count_documents({
            "user_id": current_user.id,
            "created_at": {"$gte": week_ago}
        })
        
        recent_clips = clips_collection.count_documents({
            "user_id": current_user.id,
            "created_at": {"$gte": week_ago}
        })
        
        return {
            "total_videos": total_videos,
            "total_clips": total_clips,
            "recent_videos": recent_videos,
            "recent_clips": recent_clips
        }
    except Exception as e:
        logging.error(f"Error getting user history summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/clips-info/{video_id}")
async def get_clips_info(video_id: str):
    """Get clips_info.json for a video from S3"""
    try:
        # Try to get clips_info.json from S3
        clips_info_s3_key = f"{settings.S3_UPLOAD_PREFIX}{video_id}/clips_info.json"
        
        if not s3_client.available:
            raise HTTPException(status_code=501, detail="S3 not configured")
        
        # Check if clips_info.json exists in S3
        try:
            response = s3_client.s3_client.head_object(Bucket=s3_client.bucket, Key=clips_info_s3_key)
            clips_info_s3_url = s3_client.get_object_url(clips_info_s3_key)
            
            return {
                "video_id": video_id,
                "clips_info_url": clips_info_s3_url,
                "exists": True
            }
        except Exception as e:
            # If clips_info.json doesn't exist, return empty response
            return {
                "video_id": video_id,
                "clips_info_url": None,
                "exists": False,
                "message": "clips_info.json not found in S3"
            }
            
    except Exception as e:
        logging.error(f"Error getting clips_info.json for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")










# Run the application
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
