import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from jobs import job_queue
from progress_tracker import ProgressTracker, PipelineStage
from services.content_analysis import analyze_content

# Import services
# Using AssemblyAI for better speaker diarization and sentence boundaries
from services.transcription_assemblyai import transcribe_audio
from services.video_processing import generate_thumbnail, process_video, create_clip
from services.user_video_service import update_user_video, get_user_video_by_video_id, add_clip_to_video, utc_now
from utils.s3_storage import s3_client
from utils.rapidapi_downloader import download_youtube_video_rapidapi
from utils.sieve_downloader import download_youtube_video_sieve
from utils.youtube_downloader import download_youtube_video
from logging_config import setup_logging
import tempfile

# Set up logging to worker.log file
setup_logging(log_dir="logs", log_file="worker.log", log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
upload_dir = Path("uploads")
output_dir = Path("outputs")
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Note: Removed Whisper model - now using AssemblyAI cloud service

# Initialize progress tracker
progress_tracker = ProgressTracker(job_queue)

# Global tracker for the current job being processed (for signal handling)
current_active_job_id = None




def handle_interruption(signo, frame):
    """Handle termination signals from AWS Spot or user"""
    global current_active_job_id
    if current_active_job_id:
        logger.warning(f"🛑 Termination signal {signo} received! Requeueing job {current_active_job_id}...")
        # Requeue job will mark it as 'rescheduling'
        job_queue.requeue_job(current_active_job_id)
    else:
        logger.info(f"Signal {signo} received, but no active job to requeue.")
    
    logger.info("Worker shutting down.")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, handle_interruption)
signal.signal(signal.SIGINT, handle_interruption)


async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old files in uploads and outputs directories.
    
    Args:
        max_age_hours: Maximum age in hours for files to keep (default 24 hours)
    """
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_uploads = 0
        cleaned_outputs = 0
        
        # Clean up old upload directories
        if upload_dir.exists():
            for video_dir in upload_dir.iterdir():
                if video_dir.is_dir():
                    # Check age of directory
                    dir_age = current_time - video_dir.stat().st_mtime
                    if dir_age > max_age_seconds:
                        try:
                            shutil.rmtree(video_dir)
                            cleaned_uploads += 1
                            logger.info(f"🧹 Cleaned up old upload directory (age: {dir_age/3600:.1f}h): {video_dir.name}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up old upload directory {video_dir}: {e}")
        
        # Clean up old output directories
        if output_dir.exists():
            for video_dir in output_dir.iterdir():
                if video_dir.is_dir():
                    # Check age of directory
                    dir_age = current_time - video_dir.stat().st_mtime
                    if dir_age > max_age_seconds:
                        try:
                            shutil.rmtree(video_dir)
                            cleaned_outputs += 1
                            logger.info(f"🧹 Cleaned up old output directory (age: {dir_age/3600:.1f}h): {video_dir.name}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up old output directory {video_dir}: {e}")
        
        if cleaned_uploads > 0 or cleaned_outputs > 0:
            logger.info(f"🧹 Cleanup complete: {cleaned_uploads} upload dirs, {cleaned_outputs} output dirs removed")
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {str(e)}", exc_info=True)


class ProcessingCancelledException(Exception):
    """Exception raised when processing is cancelled by user"""
    pass


async def check_if_cancelled(user_id: str, video_id: str) -> bool:
    """Check if video processing has been cancelled by checking database status.
    
    Args:
        user_id: User ID
        video_id: Video ID
        
    Returns:
        bool: True if cancelled (status is 'failed'), False otherwise
        
    Raises:
        ProcessingCancelledException: If processing has been cancelled
    """
    try:
        from services.user_video_service import get_user_video
        video = await get_user_video(user_id, video_id)
        
        if video:
            logger.debug(f"🔍 Cancellation check for video {video_id}: status={video.status}, error={video.error_message}")
            if video.status == "failed":
                error_msg = video.error_message or "Unknown reason"
                logger.info(f"🔍 Video {video_id} has failed status, error_message: '{error_msg}'")
                if "cancelled" in error_msg.lower():
                    logger.warning(f"🛑 Video {video_id} processing cancelled by user")
                    raise ProcessingCancelledException(f"Processing cancelled: {error_msg}")
                else:
                    logger.warning(f"⚠️ Video {video_id} failed but not due to cancellation: {error_msg}")
        else:
            logger.warning(f"⚠️ Video {video_id} not found for cancellation check")
            
        return False  # Not cancelled
        
    except ProcessingCancelledException:
        raise  # Re-raise cancellation exception
    except Exception as e:
        logger.warning(f"Error checking cancellation status: {e}")
        return False


async def initialize_worker():
    """Initialize the worker - verify AssemblyAI API key is configured"""
    try:
        logger.info("🚀 Initializing worker with AssemblyAI...")
        
        # Check GPU availability and log status
        try:
            import subprocess
            # Check for NVIDIA GPU
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                gpu_info = nvidia_result.stdout.strip()
                logger.info(f"🎮 GPU DETECTED: {gpu_info}")
                logger.info("✅ Hardware acceleration (NVENC) will be used for video encoding")
            else:
                logger.warning("⚠️ No NVIDIA GPU detected - using CPU for video processing")
        except FileNotFoundError:
            logger.warning("⚠️ nvidia-smi not found - no NVIDIA GPU available, using CPU")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ GPU check timed out - assuming no GPU available")
        except Exception as e:
            logger.warning(f"⚠️ GPU detection error: {e} - using CPU")
        
        # Network connectivity diagnostics
        logger.info("🌐 Running network connectivity diagnostics...")
        await run_network_diagnostics()
        
        # Verify AssemblyAI API key is configured
        from config import settings
        if not settings.ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY not configured in environment")
        
        logger.info("✅ Worker initialized successfully with AssemblyAI")
        logger.info("📝 Transcription will use cloud-based AssemblyAI service")
        
    except Exception as e:
        logger.error(f"❌ Error initializing worker: {str(e)}", exc_info=True)
        raise


async def run_network_diagnostics():
    """Run network connectivity diagnostics and log results."""
    import socket
    import aiohttp
    
    diagnostics = []
    
    # Test 1: DNS Resolution
    try:
        logger.info("   [1/5] Testing DNS resolution...")
        ip = socket.gethostbyname("google.com")
        diagnostics.append(f"✅ DNS resolution: google.com -> {ip}")
        logger.info(f"   ✅ DNS resolution: google.com -> {ip}")
    except Exception as e:
        diagnostics.append(f"❌ DNS resolution failed: {e}")
        logger.error(f"   ❌ DNS resolution failed: {e}")
    
    # Test 2: MongoDB DNS
    try:
        logger.info("   [2/5] Testing MongoDB DNS resolution...")
        ip = socket.gethostbyname("ac-2fb6zry-shard-00-00.76hczqd.mongodb.net")
        diagnostics.append(f"✅ MongoDB DNS: resolved to {ip}")
        logger.info(f"   ✅ MongoDB DNS: resolved to {ip}")
    except Exception as e:
        diagnostics.append(f"❌ MongoDB DNS failed: {e}")
        logger.error(f"   ❌ MongoDB DNS failed: {e}")
    
    # Test 3: HTTPS to Google (basic internet)
    try:
        logger.info("   [3/5] Testing HTTPS connectivity (google.com)...")
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                diagnostics.append(f"✅ HTTPS to Google: status {resp.status}")
                logger.info(f"   ✅ HTTPS to Google: status {resp.status}")
    except Exception as e:
        diagnostics.append(f"❌ HTTPS to Google failed: {type(e).__name__}: {e}")
        logger.error(f"   ❌ HTTPS to Google failed: {type(e).__name__}: {e}")
    
    # Test 4: RapidAPI endpoint
    try:
        logger.info("   [4/5] Testing RapidAPI connectivity...")
        async with aiohttp.ClientSession() as session:
            async with session.get("https://youtube-media-downloader.p.rapidapi.com/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                diagnostics.append(f"✅ RapidAPI endpoint: status {resp.status}")
                logger.info(f"   ✅ RapidAPI endpoint: status {resp.status}")
    except Exception as e:
        diagnostics.append(f"❌ RapidAPI endpoint failed: {type(e).__name__}: {e}")
        logger.error(f"   ❌ RapidAPI endpoint failed: {type(e).__name__}: {e}")
    
    # Test 5: ZylaLabs endpoint
    try:
        logger.info("   [5/5] Testing ZylaLabs connectivity...")
        async with aiohttp.ClientSession() as session:
            async with session.get("https://zylalabs.com/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                diagnostics.append(f"✅ ZylaLabs endpoint: status {resp.status}")
                logger.info(f"   ✅ ZylaLabs endpoint: status {resp.status}")
    except Exception as e:
        diagnostics.append(f"❌ ZylaLabs endpoint failed: {type(e).__name__}: {e}")
        logger.error(f"   ❌ ZylaLabs endpoint failed: {type(e).__name__}: {e}")
    
    # Summary
    logger.info("🌐 Network diagnostics complete:")
    for d in diagnostics:
        logger.info(f"   {d}")



async def process_youtube_download_job(job_id: str, job_data: dict):
    """Process YouTube download job"""
    try:
        url = job_data.get("url")
        video_id = job_data.get("video_id")
        auto_process = job_data.get("auto_process", True)

        logger.info(f"Processing YouTube download job {job_id} for URL: {url}")

        # Check if cancelled before starting
        user_id = job_data.get("user_id")
        if user_id and video_id:
            await check_if_cancelled(user_id, video_id)

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "downloading",
                "updated_at": utc_now()
            })

        # Update job status - start at 5% (auto-increments sequentially)
        progress_tracker.update_progress(job_id, PipelineStage.DOWNLOADING)
        job_queue.update_job(job_id, {"message": "Downloading video from YouTube..."})

        # Create video directory
        video_dir = upload_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Import cancellation helpers for download
        from utils.sieve_downloader import (
            set_download_cancellation_context,
            clear_download_cancellation_context,
            mark_download_cancelled
        )
        
        # Create a background task to periodically check for cancellation during download
        download_cancelled = False
        async def cancellation_monitor():
            """Background task to monitor for cancellation during download"""
            nonlocal download_cancelled
            while not download_cancelled:
                await asyncio.sleep(2)  # Check every 2 seconds
                try:
                    from services.user_video_service import get_user_video
                    video = await get_user_video(user_id, video_id)
                    if video and video.status == "failed":
                        error_msg = video.error_message or ""
                        if "cancelled" in error_msg.lower():
                            logger.warning(f"🛑 Cancellation detected during download for video {video_id}")
                            mark_download_cancelled()
                            download_cancelled = True
                            return
                except Exception as e:
                    logger.debug(f"Cancellation check error: {e}")
        
        
        # Set cancellation context
        if user_id and video_id:
            set_download_cancellation_context(user_id, video_id)

        # Download the video (RapidAPI)
        file_path, title, video_info = None, None, None
        
        # Start cancellation monitor
        monitor_task = asyncio.create_task(cancellation_monitor()) if user_id and video_id else None
        
        try:
            # Check for cancellation before download
            if user_id and video_id:
                await check_if_cancelled(user_id, video_id)
            
            # PRIMARY: RapidAPI download
            try:
                logger.info(f"Job {job_id}: [RapidAPI] Attempting download (PRIMARY)...")
                file_path, title, video_info = await download_youtube_video_rapidapi(url, video_dir)
                if file_path and title:
                    logger.info(f"Job {job_id}: ✅ RapidAPI download successful")
            except Exception as rapidapi_error:
                error_msg = str(rapidapi_error)
                if "cancelled" in error_msg.lower():
                    raise ProcessingCancelledException("Download cancelled by user")
                logger.warning(f"Job {job_id}: RapidAPI download failed: {error_msg}")
            
            # Check for cancellation
            if user_id and video_id:
                await check_if_cancelled(user_id, video_id)
        finally:
            # Stop the cancellation monitor
            download_cancelled = True
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            clear_download_cancellation_context()

        if not file_path or not title:
            raise Exception("Failed to download YouTube video using RapidAPI.")

        # Check if cancelled immediately after download
        logger.info(f"Job {job_id}: 🔍 Checking for cancellation after download...")
        if user_id and video_id:
            await check_if_cancelled(user_id, video_id)
        logger.info(f"Job {job_id}: ✅ No cancellation detected, continuing...")

        # Update MongoDB with video info
        if user_id:
            await update_user_video(user_id, video_id, {
                "title": title,
                "filename": Path(file_path).name,
                "thumbnail_url": video_info.get("thumbnail_url") if video_info else None,
                "duration": video_info.get("length_seconds") if video_info else None,
                "status": "downloaded",
                "updated_at": utc_now()
            })

        # Update job status
        progress_tracker.update_progress(job_id, PipelineStage.DOWNLOADED)
        job_queue.update_job(job_id, {"message": "YouTube video downloaded successfully",
                "video_info": json.dumps(
                    {
                        "video_id": video_id,
                        "filename": Path(file_path).name,
                        "title": title,
                        "thumbnail": video_info.get("thumbnail_url")
                        if video_info
                        else None,
                        "duration": video_info.get("length_seconds")
                        if video_info
                        else None,
                    }
                ),
            },
        )

        # Start processing if requested
        if auto_process:
            # Check if cancelled before starting processing
            if user_id and video_id:
                await check_if_cancelled(user_id, video_id)
            
            await asyncio.sleep(2)  # Small delay

            # Create a new processing job
            process_job_id = job_queue.add_job(
                "process_video",
                {
                    "video_id": video_id,
                    "file_path": file_path,
                    "original_job_id": job_id,
                    "user_id": user_id,  # Pass user_id to processing job
                },
            )

            # Update MongoDB with process_task_id so frontend can track the processing job
            logger.info(f"🔍 Updating process_task_id in database")
            logger.info(f"   user_id: {user_id}")
            logger.info(f"   video_id: {video_id}")
            logger.info(f"   process_job_id: {process_job_id}")
            
            if user_id:
                update_result = await update_user_video(user_id, video_id, {
                    "process_task_id": process_job_id,
                    "updated_at": utc_now()
                })
                if update_result:
                    logger.info(f"✅ Updated MongoDB: video {video_id} with process_task_id: {process_job_id}")
                else:
                    logger.error(f"❌ Failed to update process_task_id in MongoDB for video {video_id}")
            else:
                logger.error(f"❌ Cannot update process_task_id - user_id is None!")

            # Update original job with processing job ID
            job_queue.update_job(
                job_id,
                {
                    "status": "processing_started",
                    "message": "Processing has started",
                    "process_job_id": process_job_id,
                },
            )

            # Process the video
            await process_video_job(
                process_job_id,
                {
                    "video_id": video_id,
                    "file_path": file_path,
                    "original_job_id": job_id,
                    "user_id": user_id,  # Pass user_id to processing job
                },
            )
        
        # Release lock on successful completion
        job_queue.release_job_lock(job_id)

    except ProcessingCancelledException as e:
        logger.warning(f"YouTube download cancelled for job {job_id}: {str(e)}")
        progress_tracker.update_progress(job_id, PipelineStage.CANCELLED)
        job_queue.update_job(job_id, {"message": str(e)}
        )
        # Status already set to failed in DB by cancel endpoint
        logger.info(f"✅ Gracefully stopped YouTube download for cancelled video {video_id if 'video_id' in locals() else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error in YouTube download job {job_id}: {str(e)}", exc_info=True)
        
        # Update MongoDB video status to failed
        user_id = job_data.get("user_id")
        if user_id and 'video_id' in locals():
            await update_user_video(user_id, video_id, {
                "status": "failed",
                "error_message": str(e),
                "updated_at": utc_now()
            })
        
        # Clean up failed download files
        if 'video_id' in locals():
            try:
                video_upload_dir = upload_dir / video_id
                if video_upload_dir.exists():
                    shutil.rmtree(video_upload_dir)
                    logger.info(f"✅ Cleaned up failed download directory: {video_upload_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️  Failed to clean up directory after download error: {cleanup_error}")
        
        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error processing YouTube download: {str(e)}",
            },
        )
        
        # Release lock on error
        job_queue.release_job_lock(job_id)


async def process_video_job(job_id: str, job_data: dict):
    """Process video processing job"""
    try:
        video_id = job_data.get("video_id")
        file_path = job_data.get("file_path")
        original_job_id = job_data.get("original_job_id")
        user_id = job_data.get("user_id")
        target_clip_duration = job_data.get("target_clip_duration")  # From frontend

        # If user_id is not provided, fetch it from DB
        if not user_id:
            video_info = await get_user_video_by_video_id(video_id)
            if video_info:
                user_id = video_info.get("user_id")
            else:
                logger.error(f"Could not find user for video {video_id}")
                user_id = None

        logger.info(f"Processing video job {job_id} for video {video_id}")
        logger.info(f"   Target Clip Duration: {target_clip_duration}s")
        
        # ⏱️ START PIPELINE TIMING
        import time
        pipeline_start = time.time()
        logger.info("="*70)
        logger.info("🚀 STARTING VIDEO PROCESSING PIPELINE")
        logger.info("="*70)

        # Check if cancelled before starting
        if user_id:
            await check_if_cancelled(user_id, video_id)

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "transcribing",
                "updated_at": utc_now()
            })

        # Update job status - start at 25% (auto-increments sequentially)
        progress_tracker.update_progress(job_id, PipelineStage.TRANSCRIBING, original_job_id=original_job_id)
        job_queue.update_job(job_id, {"message": "Scanning audio for content"})

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at path: {file_path}")

        # ⏱️ STEP 1: TRANSCRIPTION
        step_start = time.time()
        logger.info("📝 STEP 1: Starting transcription (AssemblyAI + Speaker Diarization)...")
        
        
        # AssemblyAI handles transcription in the cloud - no local model needed
        try:
            file_size = None
            try:
                file_size = os.path.getsize(file_path)
                logger.info(f"   Transcription input size: {file_size / (1024*1024):.2f} MB")
            except Exception:
                logger.info("   Transcription input size: unknown")

            t_trans_start = time.time()
            transcript = await transcribe_audio(file_path)
            t_trans_end = time.time()
            logger.info(f"   transcribe_audio() total time: {t_trans_end - t_trans_start:.2f}s")
            
            # Check if cancelled immediately after transcription
            if user_id:
                await check_if_cancelled(user_id, video_id)
                
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            raise
        finally:
            pass
        
        if not transcript:
            raise ValueError("Transcription returned None")
        
        step_elapsed = time.time() - step_start
        logger.info(f"✅ STEP 1 COMPLETE: Transcription finished in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
        logger.info(f"   - Total segments: {len(transcript.get('segments', []))}")
        logger.info(f"   - Total sentences: {len(transcript.get('sentences', []))}")
        logger.info(f"   - Speakers detected: {len(transcript.get('speakers', []))}")
        logger.info(f"   - Has speaker labels: {'Yes' if transcript.get('transcript_for_ai') else 'No'}")

        # Handle empty segments gracefully - this is valid for silent/corrupted videos
        segments_count = len(transcript.get("segments", []))
        logger.info(f"Transcription completed with {segments_count} segments")

        if segments_count == 0:
            logger.warning("Video appears to be silent or have no speech content")
            # Continue processing with empty transcript - don't fail

        # Check if cancelled before analysis
        if user_id:
            await check_if_cancelled(user_id, video_id)

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "analyzing",
                "updated_at": utc_now()
            })

        # Update job status - jump to 50% (auto-increments sequentially)
        progress_tracker.update_progress(job_id, PipelineStage.ANALYZING, original_job_id=original_job_id)
        job_queue.update_job(job_id, {"message": "Analyzing content"})

        # ⏱️ STEP 2: AI CONTENT ANALYSIS
        step_start = time.time()
        logger.info("🤖 STEP 2: Starting AI content analysis (OpenAI clip detection)...")
        logger.info(f"   Target clip duration: {target_clip_duration}s")
        
        
        try:
            segments = await analyze_content(transcript, target_clip_duration=target_clip_duration)
            
            # Check if cancelled immediately after analysis
            if user_id:
                await check_if_cancelled(user_id, video_id)
        finally:
            # Stop the progress simulation
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        if not segments:
            logger.warning(
                "Content analysis found no interesting segments, using default"
            )
            segments = [
                {
                    "start_time": 0,
                    "end_time": 60,
                    "title": "Automatic Clip",
                    "description": "Automatically generated clip",
                }
            ]
        
        step_elapsed = time.time() - step_start
        logger.info(f"✅ STEP 2 COMPLETE: AI analysis finished in {step_elapsed:.2f} seconds")
        logger.info(f"   - Clips identified: {len(segments)}")

        # Check if cancelled before video processing
        if user_id:
            await check_if_cancelled(user_id, video_id)

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "processing",
                "updated_at": utc_now()
            })

        # Update job status - jump to 70% (auto-increments sequentially)
        progress_tracker.update_progress(job_id, PipelineStage.PROCESSING, original_job_id=original_job_id)
        job_queue.update_job(job_id, {"message": "Processing video clips"})

        # Create output directory
        clips_dir = output_dir / video_id
        clips_dir.mkdir(exist_ok=True)

        # ⏱️ STEP 3: VIDEO PROCESSING
        step_start = time.time()
        logger.info("🎬 STEP 3: Starting video clip creation (FFmpeg processing)...")
        
        
        # Set cancellation context for nested processing functions
        from services.video_processing import set_cancellation_context, clear_cancellation_context
        if user_id:
            set_cancellation_context(user_id, video_id)
        
        try:
            clips = await process_video(file_path, transcript, segments)
        finally:
            # Always clear the cancellation context
            clear_cancellation_context()
        
        # Check if cancelled immediately after video processing
        if user_id:
            await check_if_cancelled(user_id, video_id)
        
        step_elapsed = time.time() - step_start
        logger.info(f"✅ STEP 3 COMPLETE: Video processing finished in {step_elapsed:.2f} seconds")
        logger.info(f"   - Clips created: {len(clips) if clips else 0}")

        # ====================================================================
        # STEP 4: PARALLEL CLIP PROCESSING - S3 upload and thumbnail generation
        # ====================================================================
        
        if clips and s3_client.available:
            logger.info(f"[PARALLEL] Starting parallel processing of {len(clips)} clips")
            
            parallel_start_time = time.time()
            
            # Prepare for parallel processing
            MAX_CONCURRENT_CLIPS = 3  # Adjust based on system resources
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)
            
            async def bounded_process_clip(clip, idx):
                """Process clip with concurrency limit"""
                async with semaphore:
                    return await process_single_clip_async(
                        clip, idx, file_path, video_id, output_dir, transcript
                    )
            
            # Update progress - starting parallel processing
            job_queue.update_job(
                job_id,
                {
                    "progress": 75,
                    "message": f"Uploading {len(clips)} clips to S3 in parallel...",
                },
            )
            
            if original_job_id:
                job_queue.update_job(
                    original_job_id,
                    {
                        "progress": 75,
                        "message": f"Uploading {len(clips)} clips to S3 in parallel...",
                    },
                )
            
            # Create all clip processing tasks
            clip_tasks = [
                bounded_process_clip(clip, i)
                for i, clip in enumerate(clips)
            ]
            
            # Run all tasks in parallel
            logger.info(f"[PARALLEL] Running {len(clip_tasks)} clips in parallel (max {MAX_CONCURRENT_CLIPS} concurrent)")
            results = await asyncio.gather(*clip_tasks, return_exceptions=True)
            
            # Process results
            logger.info(f"[PARALLEL] All clips processed, updating results...")
            for result in results:
                if result and isinstance(result, tuple) and len(result) == 2:
                    index, data = result
                    if data and index < len(clips):
                        clips[index]['s3_key'] = data.get('clip_s3_key')
                        clips[index]['s3_url'] = s3_client.get_object_url(data['clip_s3_key']) if data.get('clip_s3_key') else None
                        clips[index]['thumbnail_s3_key'] = data.get('thumbnail_s3_key')
                        clips[index]['thumbnail_url'] = s3_client.get_object_url(data['thumbnail_s3_key']) if data.get('thumbnail_s3_key') else "/static/default_thumbnail.jpg"
                        logger.info(f"[PARALLEL] ✅ Clip {index} complete - S3 URL: {clips[index].get('s3_url')}")
                    else:
                        logger.warning(f"[PARALLEL] ❌ Clip {index} returned None results")
                        if index < len(clips):
                            clips[index]['thumbnail_url'] = "/static/default_thumbnail.jpg"
                elif isinstance(result, Exception):
                    logger.error(f"[PARALLEL] ❌ Clip raised exception: {result}")
                else:
                    logger.warning(f"[PARALLEL] ❌ Clip returned invalid result type")
            
            parallel_elapsed = time.time() - parallel_start_time
            logger.info(f"[PARALLEL] ⏱️  Total time: {parallel_elapsed:.1f}s")
            logger.info(f"[PARALLEL] 📊 Average per clip: {parallel_elapsed/len(clips):.1f}s")
            if len(clips) > 1:
                sequential_estimate = parallel_elapsed * len(clips) / MAX_CONCURRENT_CLIPS
                speedup = sequential_estimate / parallel_elapsed if parallel_elapsed > 0 else 0
                logger.info(f"[PARALLEL] 📈 Estimated speedup vs sequential: ~{speedup:.1f}x")
            
            logger.info(f"[PARALLEL] ✅ Parallel processing complete for all {len(clips)} clips")
        else:
            if not clips:
                logger.info("No clips to process")
            elif not s3_client.available:
                logger.warning("S3 client not available, skipping clip uploads")

        # ====================================================================
        # STEP 5: SAVE CLIPS TO DATABASE
        # ====================================================================
        
        if clips and user_id:
            logger.info(f"💾 STEP 5: Saving {len(clips)} clips to database...")
            logger.info(f"🔍 CLIP SAVE CONTEXT:")
            logger.info(f"   User ID: {user_id}")
            logger.info(f"   Video ID: {video_id}")
            logger.info(f"   Number of clips: {len(clips)}")
            
            # Verify video exists before saving clips
            from services.user_video_service import get_user_video
            verify_video = await get_user_video(user_id, video_id)
            if verify_video:
                logger.info(f"✅ Video found in DB before saving clips")
                logger.info(f"   Video process_task_id: {verify_video.process_task_id}")
                logger.info(f"   Current clips count: {len(verify_video.clips)}")
            else:
                logger.error(f"❌ Video {video_id} NOT FOUND in user {user_id}'s videos before saving clips!")
            
            clips_saved = 0
            
            for i, clip in enumerate(clips):
                if clip.get("s3_url"):  # Only save clips with S3 URLs
                    clip_data = {
                        "title": clip.get("title", f"Clip {i+1}"),
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
                    
                    try:
                        logger.info(f"📝 Attempting to save clip {i+1}/{len(clips)}...")
                        clip_id = await add_clip_to_video(user_id, video_id, clip_data)
                        if clip_id:
                            logger.info(f"✅ Saved clip {i+1} to database with ID: {clip_id}")
                            logger.info(f"   Clip title: {clip.get('title')}")
                            logger.info(f"   S3 URL: {clip.get('s3_url')[:50]}..." if clip.get('s3_url') else "   S3 URL: None")
                            clips_saved += 1
                        else:
                            logger.error(f"❌ Failed to save clip {i+1} to database - add_clip_to_video returned None")
                            logger.error(f"   User ID: {user_id}")
                            logger.error(f"   Video ID: {video_id}")
                    except Exception as clip_save_error:
                        logger.error(f"❌ Error saving clip {i+1}: {clip_save_error}")
                        import traceback
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                else:
                    logger.warning(f"⚠️ Skipping clip {i+1} - no S3 URL")
            
            logger.info(f"✅ Saved {clips_saved}/{len(clips)} clips to database")
            
            # Final verification - check if clips were actually saved
            if clips_saved > 0:
                from services.user_video_service import get_user_video
                final_video = await get_user_video(user_id, video_id)
                if final_video:
                    logger.info(f"🔍 FINAL VERIFICATION:")
                    logger.info(f"   Video ID: {video_id}")
                    logger.info(f"   Process Task ID: {final_video.process_task_id}")
                    logger.info(f"   Clips in DB: {len(final_video.clips)}")
                    logger.info(f"   Expected clips: {clips_saved}")
                    if len(final_video.clips) != clips_saved:
                        logger.error(f"❌ MISMATCH: Expected {clips_saved} clips but found {len(final_video.clips)} in DB!")
                    else:
                        logger.info(f"✅ Clip count matches - all clips saved successfully")
                else:
                    logger.error(f"❌ Could not verify - video {video_id} not found after saving clips")

        # Save metadata
        metadata = {"video_id": video_id, "clips": clips}
        metadata_path = clips_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # ⏱️ PIPELINE COMPLETE - TOTAL TIME
        total_elapsed = time.time() - pipeline_start
        logger.info("="*70)
        logger.info("🎉 PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"⏱️  TOTAL PROCESSING TIME: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        logger.info(f"📊 SUMMARY:")
        logger.info(f"   - Video ID: {video_id}")
        logger.info(f"   - Clips created: {len(clips)}")
        logger.info(f"   - Transcript segments: {len(transcript.get('segments', []))}")
        logger.info("="*70)

        # Update job status
        progress_tracker.update_progress(job_id, PipelineStage.COMPLETED)
        job_queue.update_job(job_id, {"message": "Processing completed",
                "clips": json.dumps(clips),
            },
        )

        if original_job_id:
            progress_tracker.update_progress(original_job_id, PipelineStage.COMPLETED)
            job_queue.update_job(original_job_id, {"message": "Processing completed",
                    "clips": json.dumps(clips),
                },
            )

        # After successful processing, update video document
        if user_id:
            # Find a representative thumbnail for the video (first clip with a thumbnail)
            video_thumbnail_url = None
            for clip in clips:
                if clip.get("thumbnail_url") and not clip["thumbnail_url"].endswith("default_thumbnail.jpg"):
                    video_thumbnail_url = clip["thumbnail_url"]
                    break
            # Set processed_at and status
            await update_user_video(user_id, video_id, {
                "status": "completed",
                "processed_at": utc_now(),
                "thumbnail_url": video_thumbnail_url,
                "clip_thumbnail_url": video_thumbnail_url,
                # Optionally add s3_url, etc. if you have them
            })

        # Clean up all local files immediately after successful processing
        # (Don't wait 24 hours - original videos aren't needed after clips are generated)
        cleanup_success = True
        try:
            # Clean up uploads directory (original video)
            video_upload_dir = upload_dir / video_id
            if video_upload_dir.exists():
                dir_size_mb = sum(f.stat().st_size for f in video_upload_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                shutil.rmtree(video_upload_dir)
                logger.info(f"🗑️ Deleted original video directory: {video_upload_dir} ({dir_size_mb:.1f}MB freed)")
            
            # Clean up outputs directory (local clips - already uploaded to S3)
            video_output_dir = output_dir / video_id
            if video_output_dir.exists():
                dir_size_mb = sum(f.stat().st_size for f in video_output_dir.rglob('*') if f.is_file()) / (1024 * 1024)
                shutil.rmtree(video_output_dir)
                logger.info(f"🗑️ Deleted output directory: {video_output_dir} ({dir_size_mb:.1f}MB freed)")
                
        except Exception as cleanup_error:
            cleanup_success = False
            logger.warning(f"⚠️ Failed to clean up directories: {cleanup_error}")
        
        if cleanup_success:
            logger.info("✅ Immediate cleanup complete - all local files deleted")

        logger.info(f"Job {job_id} completed successfully with {len(clips)} clips")
        
        # Release lock on successful completion
        job_queue.release_job_lock(job_id)
        if original_job_id:
            job_queue.release_job_lock(original_job_id)
        
        # Delete the completed job from Redis to prevent accumulation
        job_queue.delete_job(job_id)
        if original_job_id:
            job_queue.delete_job(original_job_id)

    except ProcessingCancelledException as e:
        logger.warning(f"Processing cancelled for job {job_id}: {str(e)}")
        progress_tracker.update_progress(job_id, PipelineStage.CANCELLED)
        job_queue.update_job(job_id, {"message": str(e)}
        )
        if original_job_id:
            progress_tracker.update_progress(original_job_id, PipelineStage.CANCELLED)
            job_queue.update_job(original_job_id, {"message": str(e)})
        # Status already set to failed in DB by cancel endpoint
        logger.info(f"✅ Gracefully stopped processing for cancelled video {video_id if 'video_id' in locals() else 'unknown'}")
        # Clean up any partial files
        if 'video_id' in locals():
            try:
                video_upload_dir = upload_dir / video_id
                if video_upload_dir.exists():
                    shutil.rmtree(video_upload_dir)
                    logger.info(f"🗑️ Cleaned up cancelled upload directory: {video_upload_dir}")
                video_output_dir = output_dir / video_id
                if video_output_dir.exists():
                    shutil.rmtree(video_output_dir)
                    logger.info(f"🗑️ Cleaned up cancelled output directory: {video_output_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up cancelled files: {cleanup_error}")
        
        # Delete cancelled job from Redis
        job_queue.delete_job(job_id)
        if original_job_id:
            job_queue.delete_job(original_job_id)
    
    # Also catch ProcessingCancelledException from video_processing module
    except Exception as e:
        # Check if it's a cancellation exception from video_processing module
        from services.video_processing import ProcessingCancelledException as VPCancelledException
        if isinstance(e, VPCancelledException):
            logger.warning(f"Processing cancelled (from video_processing) for job {job_id}: {str(e)}")
            progress_tracker.update_progress(job_id, PipelineStage.CANCELLED)
            job_queue.update_job(job_id, {"message": str(e)})
            if original_job_id:
                progress_tracker.update_progress(original_job_id, PipelineStage.CANCELLED)
                job_queue.update_job(original_job_id, {"message": str(e)})
            logger.info(f"✅ Gracefully stopped processing for cancelled video {video_id if 'video_id' in locals() else 'unknown'}")
            # Clean up any partial files
            if 'video_id' in locals():
                try:
                    video_upload_dir = upload_dir / video_id
                    if video_upload_dir.exists():
                        shutil.rmtree(video_upload_dir)
                    video_output_dir = output_dir / video_id
                    if video_output_dir.exists():
                        shutil.rmtree(video_output_dir)
                except Exception:
                    pass
            return
        
        logger.error(f"Error in video processing job {job_id}: {str(e)}", exc_info=True)
        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error: {str(e)}"}
        )

        if original_job_id:
            progress_tracker.update_progress(original_job_id, PipelineStage.ERROR)
            job_queue.update_job(original_job_id, {"message": f"Error: {str(e)}"})
        # Update video status to failed in DB
        if 'video_id' in locals():
            # Fetch user_id if not already available
            if not user_id:
                video_info = await get_user_video_by_video_id(video_id)
                if video_info:
                    user_id = video_info.get("user_id")
            if user_id:
                await update_user_video(user_id, video_id, {
                    "status": "failed",
                    "error_message": str(e),
                    "processed_at": utc_now(),
                })
            
            # Clean up failed processing files
            try:
                # Clean up uploads directory
                video_upload_dir = upload_dir / video_id
                if video_upload_dir.exists():
                    shutil.rmtree(video_upload_dir)
                    logger.info(f"✅ Cleaned up upload directory after processing failure: {video_upload_dir}")
                
                # Clean up outputs directory
                video_output_dir = output_dir / video_id
                if video_output_dir.exists():
                    shutil.rmtree(video_output_dir)
                    logger.info(f"✅ Cleaned up output directory after processing failure: {video_output_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️  Failed to clean up directories after processing error: {cleanup_error}")
        
        # Release lock on error
        job_queue.release_job_lock(job_id)
        if original_job_id:
            job_queue.release_job_lock(original_job_id)
        
        # Delete failed job from Redis
        job_queue.delete_job(job_id)
        if original_job_id:
            job_queue.delete_job(original_job_id)


async def process_manual_clip_job(job_id: str, job_data: dict):
    """Process manual clip generation job"""
    try:
        video_id = job_data.get("video_id")
        video_path = job_data.get("video_path")
        start_time = float(job_data.get("start_time"))
        end_time = float(job_data.get("end_time"))
        title = job_data.get("title", "Manual Clip")

        logger.info(f"Processing manual clip job {job_id} for video {video_id}")

        # Update job status using progress tracker
        progress_tracker.update_progress_explicit(job_id, "processing", 20)
        job_queue.update_job(job_id, {"message": "Processing clip"})

        # Create output directory
        clips_dir = output_dir / video_id
        clips_dir.mkdir(exist_ok=True)

        # Create manual segment
        manual_segment = {
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "description": "Manually created clip",
        }

        # Minimal transcript for manual clips
        minimal_transcript = {"text": "", "segments": []}

        # Process the video clip
        clips = await process_video(video_path, minimal_transcript, [manual_segment])

        if not clips or len(clips) == 0:
            raise Exception("Failed to generate clip")

        clip = clips[0]

        # Generate thumbnail
        clip_path = clip.get("path")
        if clip_path and os.path.exists(clip_path):
            thumbnail_filename = f"{clip.get('id')}_thumbnail.jpg"
            thumbnail_path = os.path.abspath(str(clips_dir / thumbnail_filename))

            clip_duration = end_time - start_time
            thumbnail_timestamp = (
                min(clip_duration / 2, clip_duration - 0.5)
                if clip_duration > 1.0
                else 0
            )

            try:
                generated_path = await generate_thumbnail(
                    clip_path, thumbnail_path, thumbnail_timestamp
                )

                if (
                    generated_path
                    and os.path.exists(thumbnail_path)
                    and os.path.getsize(thumbnail_path) > 0
                ):
                    clip["thumbnail_path"] = thumbnail_path
                    clip["thumbnail_url"] = f"/outputs/{video_id}/{thumbnail_filename}"
                else:
                    clip["thumbnail_url"] = "/static/default_thumbnail.jpg"
            except Exception as thumb_error:
                logger.error(f"Error generating thumbnail: {str(thumb_error)}")
                clip["thumbnail_url"] = "/static/default_thumbnail.jpg"
        else:
            clip["thumbnail_url"] = "/static/default_thumbnail.jpg"

        # Update job status
        progress_tracker.update_progress(job_id, PipelineStage.COMPLETED)
        job_queue.update_job(job_id, {"message": "Clip generated successfully",
                "clip": json.dumps(clip),
            },
        )

        logger.info(f"Manual clip job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error in manual clip job {job_id}: {str(e)}", exc_info=True)
        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error: {str(e)}"}
        )


async def process_s3_download_job(job_id: str, job_data: dict):
    """Process S3 download and process job"""
    try:
        video_id = job_data.get("video_id")
        s3_key = job_data.get("s3_key")
        local_path = job_data.get("local_path")

        logger.info(f"Processing S3 download job {job_id} for video {video_id}")

        # Update job status
        progress_tracker.update_progress_explicit(job_id, "downloading", 5)
        job_queue.update_job(job_id, {"message": "Downloading file from S3..."})

        # Download from S3
        success = s3_client.download_from_s3(s3_key, local_path)

        if not success:
            raise Exception("Failed to download file from S3")

        # Update job status
        progress_tracker.update_progress_explicit(job_id, "downloaded", 10)
        job_queue.update_job(job_id, {"message": "File downloaded from S3, starting processing"})

        # Now process the video like a regular processing job
        await process_video_job(job_id, {"video_id": video_id, "file_path": local_path})

    except Exception as e:
        logger.error(f"Error in S3 download job {job_id}: {str(e)}", exc_info=True)
        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error downloading from S3: {str(e)}",
            },
        )


async def _upload_clip_to_s3_async(clip_path: str, video_id: str, clip_id: str) -> tuple:
    """Upload clip to S3 asynchronously"""
    try:
        if not os.path.exists(clip_path):
            logger.error(f"Clip file not found: {clip_path}")
            return False, None
            
        loop = asyncio.get_running_loop()  # Use get_running_loop() instead of deprecated get_event_loop()
        file_size = os.path.getsize(clip_path)
        
        logger.info(f"[S3 Upload] Starting upload: {clip_path} ({file_size/1024/1024:.1f}MB)")
        
        # Use multipart for large files
        if file_size > 50 * 1024 * 1024:  # 50MB
            success, s3_key = await loop.run_in_executor(
                None,
                lambda: s3_client.upload_clip_to_s3_multipart(
                    clip_path, video_id, f"clip_{clip_id}.mp4"
                )
            )
        else:
            success, s3_key = await loop.run_in_executor(
                None,
                lambda: s3_client.upload_clip_to_s3(
                    clip_path, video_id, f"clip_{clip_id}.mp4"
                )
            )
        
        logger.info(f"[S3 Upload] Completed: success={success}, key={s3_key}")
        return success, s3_key
    except Exception as e:
        logger.error(f"Error uploading clip to S3: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None


async def _upload_thumbnail_to_s3_async(thumbnail_path: str, video_id: str, clip_id: str) -> tuple:
    """Upload thumbnail to S3 asynchronously"""
    try:
        if not os.path.exists(thumbnail_path):
            logger.warning(f"Thumbnail file not found: {thumbnail_path}")
            return False, None
            
        loop = asyncio.get_running_loop()  # Use get_running_loop() instead of deprecated get_event_loop()
        success, s3_key = await loop.run_in_executor(
            None,
            lambda: s3_client.upload_thumbnail_to_s3(thumbnail_path, video_id, f"clip_{clip_id}_thumbnail.jpg")
        )
        return success, s3_key
    except Exception as e:
        logger.error(f"Error uploading thumbnail to S3: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None


async def process_single_clip_async(
    clip: dict,
    index: int,
    file_path: str,
    video_id: str,
    output_dir_path: Path,
    transcript: dict = None
) -> tuple:
    """Upload existing clip and thumbnail to S3 (clips already created in process_video)"""
    try:
        logger.info(f"[PARALLEL] Starting S3 upload for clip {index}: {clip.get('id')}")
        
        # Use the already-created clip from process_video() - NO RE-PROCESSING
        created_clip_path = clip.get('path')
        thumbnail_path = clip.get('thumbnail_path')
        
        if not created_clip_path or not os.path.exists(created_clip_path):
            logger.warning(f"[PARALLEL] Clip {index} path not found: {created_clip_path}")
            return index, None
        
        logger.info(f"[PARALLEL] Using existing clip at {created_clip_path}")
        
        # Verify thumbnail exists
        if thumbnail_path and os.path.exists(thumbnail_path):
            logger.info(f"[PARALLEL] Using existing thumbnail at {thumbnail_path}")
        else:
            logger.warning(f"[PARALLEL] Thumbnail not found at {thumbnail_path}")
            thumbnail_path = None
        
        # Upload to S3 (clip and thumbnail in parallel)
        clip_upload = asyncio.create_task(
            _upload_clip_to_s3_async(created_clip_path, video_id, clip.get("id"))
        )
        
        thumbnail_upload = None
        if thumbnail_path and os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
            thumbnail_upload = asyncio.create_task(
                _upload_thumbnail_to_s3_async(thumbnail_path, video_id, clip.get("id"))
            )
        
        # Wait for uploads
        clip_result = await clip_upload
        thumb_result = await thumbnail_upload if thumbnail_upload else (False, None)
        
        logger.info(f"[PARALLEL] Clip {index} S3 upload: clip={clip_result[0]}, thumb={thumb_result[0]}")
        
        # Cleanup local files (they're in temp directory from process_video)
        try:
            if os.path.exists(created_clip_path):
                os.unlink(created_clip_path)
            if thumbnail_path and os.path.exists(thumbnail_path):
                os.unlink(thumbnail_path)
        except Exception as cleanup_err:
            logger.warning(f"[PARALLEL] Cleanup error for clip {index}: {cleanup_err}")
        
        # Return results
        return index, {
            'clip_s3_key': clip_result[1],
            'thumbnail_s3_key': thumb_result[1],
            'title': clip.get('title'),
            'start_time': clip.get('start_time'),
            'end_time': clip.get('end_time'),
            'transcription': clip.get('transcription', ''),
            'summary': clip.get('summary', ''),
            'tags': clip.get('tags', []),
        }
        
    except Exception as e:
        logger.error(f"[PARALLEL] Error processing clip {index}: {str(e)}")
        import traceback
        logger.error(f"[PARALLEL] Traceback: {traceback.format_exc()}")
        return index, None


async def process_uploaded_video_job(job_id: str, job_data: dict):
    """Process uploaded video from S3 - matches YouTube download flow"""
    video_id = None
    user_id = None
    local_path = None
    monitor_task = None
    processing_cancelled = False
    
    try:
        video_id = job_data.get("video_id")
        s3_key = job_data.get("s3_key")
        user_id = job_data.get("user_id")
        target_clip_duration = job_data.get("target_clip_duration", 60)

        logger.info(f"Processing uploaded video job {job_id} for video {video_id}")
        logger.info(f"   Job ID (task_id): {job_id}")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   S3 Key: {s3_key}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Target Clip Duration: {target_clip_duration}s")

        # Check if cancelled before starting
        if user_id and video_id:
            await check_if_cancelled(user_id, video_id)

        # Update MongoDB video status (matching YouTube flow)
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "downloading",
                "updated_at": utc_now()
            })

        # Update job status
        progress_tracker.update_progress_explicit(job_id, "downloading", 5)
        job_queue.update_job(job_id, {"message": "Downloading file from S3..."})

        # Create a background task to periodically check for cancellation (matching YouTube flow)
        async def cancellation_monitor():
            """Background task to monitor for cancellation during processing"""
            nonlocal processing_cancelled
            while not processing_cancelled:
                await asyncio.sleep(2)  # Check every 2 seconds
                try:
                    from services.user_video_service import get_user_video
                    video = await get_user_video(user_id, video_id)
                    if video and video.status == "failed":
                        error_msg = video.error_message or ""
                        if "cancelled" in error_msg.lower():
                            logger.warning(f"🛑 Cancellation detected during upload processing for video {video_id}")
                            processing_cancelled = True
                            return
                except Exception as e:
                    logger.debug(f"Cancellation check error: {e}")

        # Start cancellation monitor
        monitor_task = asyncio.create_task(cancellation_monitor()) if user_id and video_id else None

        try:
            # Create temporary file for processing
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                local_path = temp_file.name

            # Download from S3
            success = s3_client.download_from_s3(s3_key, local_path)

            if not success:
                raise Exception("Failed to download file from S3")

            # Check if cancelled immediately after S3 download
            if user_id and video_id:
                await check_if_cancelled(user_id, video_id)
            if processing_cancelled:
                raise ProcessingCancelledException("Processing cancelled by user")

            # Update MongoDB status
            if user_id:
                await update_user_video(user_id, video_id, {
                    "status": "processing",
                    "updated_at": utc_now()
                })

            # Update job status
            progress_tracker.update_progress_explicit(job_id, "processing", 10)
            job_queue.update_job(job_id, {"message": "File downloaded, starting processing..."})

            # Process the video using the same pipeline as process_video_job
            await process_video_job(job_id, {
                "video_id": video_id,
                "file_path": local_path,
                "user_id": user_id,
                "target_clip_duration": target_clip_duration
            })
            
            # Release lock on successful completion
            job_queue.release_job_lock(job_id)
            # Delete completed job from Redis
            job_queue.delete_job(job_id)
            
        finally:
            # Stop the cancellation monitor
            processing_cancelled = True
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    except ProcessingCancelledException as e:
        logger.warning(f"Uploaded video processing cancelled for job {job_id}: {str(e)}")
        progress_tracker.update_progress(job_id, PipelineStage.CANCELLED)
        job_queue.update_job(job_id, {"message": str(e)}
        )
        # Release lock on cancellation
        job_queue.release_job_lock(job_id)
        # Delete cancelled job from Redis
        job_queue.delete_job(job_id)
        logger.info(f"✅ Gracefully stopped uploaded video processing for cancelled video {video_id or 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error in uploaded video job {job_id}: {str(e)}", exc_info=True)
        
        # Update MongoDB video status to failed
        if user_id and video_id:
            await update_user_video(user_id, video_id, {
                "status": "failed",
                "error_message": str(e),
                "updated_at": utc_now()
            })
        
        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error processing uploaded video: {str(e)}",
            },
        )
        # Release lock on error
        job_queue.release_job_lock(job_id)
        # Delete failed job from Redis
        job_queue.delete_job(job_id)
        
    finally:
        # Clean up temporary file
        try:
            if local_path and os.path.exists(local_path):
                os.unlink(local_path)
        except:
            pass


async def process_youtube_download_job_v2(job_id: str, job_data: dict):
    """Process YouTube download with three-tier fallback system"""
    try:
        video_id = job_data.get("video_id")
        url = job_data.get("url")
        auto_process = job_data.get("auto_process", True)
        user_id = job_data.get("user_id")
        target_clip_duration = job_data.get("target_clip_duration")  # From frontend

        logger.info(f"Processing YouTube download job {job_id} for URL: {url}")
        logger.info(f"   Job ID (task_id): {job_id}")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Target Clip Duration: {target_clip_duration}s")

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "downloading",
                "updated_at": utc_now()
            })

        # Update job status
        progress_tracker.update_progress(job_id, PipelineStage.DOWNLOADING)
        job_queue.update_job(job_id, {"message": "Downloading video from YouTube..."})

        # Create video directory
        video_dir = upload_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Download the video (RapidAPI)
        file_path, title, video_info = None, None, None

        # PRIMARY: RapidAPI download
        try:
            logger.info(f"Job {job_id}: [RapidAPI] Attempting download (PRIMARY)...")
            file_path, title, video_info = await download_youtube_video_rapidapi(url, video_dir)
            if file_path and title:
                logger.info(f"Job {job_id}: ✅ RapidAPI download successful")
        except Exception as rapidapi_error:
            logger.warning(f"Job {job_id}: RapidAPI download failed: {str(rapidapi_error)}")

        if not file_path or not title:
            raise Exception("Failed to download YouTube video using RapidAPI.")

        logger.info(f"Job {job_id}: Download completed successfully: {file_path}")

        # ⚠️ CHECK FOR CANCELLATION BEFORE updating status (critical!)
        # If user cancelled during download, the video status is already "failed"
        # We must check this BEFORE overwriting with "downloaded"
        logger.info(f"Job {job_id}: 🔍 Checking for cancellation after download...")
        if user_id and video_id:
            await check_if_cancelled(user_id, video_id)
        logger.info(f"Job {job_id}: ✅ No cancellation detected, continuing...")

        # Update video in database
        if user_id:
            await update_user_video(user_id, video_id, {
                "title": title or "YouTube Video",
                "filename": Path(file_path).name,
                "source_url": url,
                "status": "downloaded",
                "video_type": "youtube",
                "duration": video_info.get("length_seconds") if video_info else None,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else None,
                "content_type": "video/mp4",
                "updated_at": utc_now()
            })

        # Update job status
        progress_tracker.update_progress(job_id, PipelineStage.DOWNLOADED)
        job_queue.update_job(job_id, {"message": "YouTube video downloaded successfully",
            },
        )

        # Start processing if requested
        if auto_process:
            await asyncio.sleep(2)

            logger.info(f"🔄 Continuing with same job ID for processing: {job_id}")

            # Update job status to indicate processing is starting
            job_queue.update_job(
                job_id,
                {
                    "status": "processing_started",
                    "message": "Processing has started",
                },
            )

            # Process the video using the same job_id
            await process_video_job(
                job_id,
                {
                    "video_id": video_id,
                    "file_path": file_path,
                    "user_id": user_id,
                    "target_clip_duration": target_clip_duration,
                },
            )
            
            # Lock already released in process_video_job

    except ProcessingCancelledException as e:
        logger.warning(f"YouTube download cancelled for job {job_id}: {str(e)}")
        progress_tracker.update_progress(job_id, PipelineStage.CANCELLED)
        job_queue.update_job(job_id, {"message": str(e)}
        )
        # Status already set to failed in DB by cancel endpoint
        logger.info(f"✅ Gracefully stopped YouTube download for cancelled video {video_id if 'video_id' in locals() else 'unknown'}")
        
        # Clean up partial download files
        if 'video_id' in locals():
            try:
                video_upload_dir = upload_dir / video_id
                if video_upload_dir.exists():
                    shutil.rmtree(video_upload_dir)
                    logger.info(f"🗑️ Cleaned up cancelled download directory: {video_upload_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up cancelled files: {cleanup_error}")
        
        # Release lock
        job_queue.release_job_lock(job_id)

    except Exception as e:
        logger.error(f"Error in YouTube download job {job_id}: {str(e)}", exc_info=True)

        # Update MongoDB video status to failed
        user_id = job_data.get("user_id")
        if user_id and 'video_id' in locals():
            await update_user_video(user_id, video_id, {
                "status": "failed",
                "error_message": str(e),
                "updated_at": utc_now()
            })

        progress_tracker.update_progress(job_id, PipelineStage.ERROR)
        job_queue.update_job(job_id, {"message": f"Error: {str(e)}",
            },
        )

        # Clean up failed download files
        if 'video_id' in locals():
            try:
                video_upload_dir = upload_dir / video_id
                if video_upload_dir.exists():
                    shutil.rmtree(video_upload_dir)
                    logger.info(f"✅ Cleaned up failed YouTube download directory: {video_upload_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up directory: {cleanup_error}")
        
        # Release lock on error
        job_queue.release_job_lock(job_id)


async def worker_main():
    """Main worker loop - supports concurrent job processing"""
    logger.info("Starting worker...")

    # Initialize worker
    await initialize_worker()
    
    # Configure max concurrent jobs (default 3 for GPU memory management)
    max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
    logger.info(f"🚀 Worker configured for {max_concurrent_jobs} concurrent jobs")

    logger.info("Worker ready, waiting for jobs...")
    
    # Track last cleanup time
    import time
    last_cleanup_time = time.time()
    cleanup_interval = 3600  # Run cleanup every hour
    
    # Semaphore to limit concurrent jobs
    job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
    
    # Track active job tasks
    active_tasks = set()
    
    async def process_job_with_semaphore(job_id: str, job_type: str, job_data: dict):
        """Process a job with semaphore for concurrency control"""
        async with job_semaphore:
            try:
                logger.info(f"🎬 Starting job {job_id} of type {job_type} (active jobs: {max_concurrent_jobs - job_semaphore._value + 1}/{max_concurrent_jobs})")
                
                # Process job based on type
                if job_type == "youtube_download":
                    await process_youtube_download_job(job_id, job_data)
                elif job_type == "process_youtube_download":
                    await process_youtube_download_job_v2(job_id, job_data)
                elif job_type == "process_uploaded_video":
                    await process_uploaded_video_job(job_id, job_data)
                elif job_type == "process_video":
                    await process_video_job(job_id, job_data)
                elif job_type == "manual_clip":
                    await process_manual_clip_job(job_id, job_data)
                elif job_type == "s3_download_and_process":
                    await process_s3_download_job(job_id, job_data)
                else:
                    logger.error(f"Unknown job type: {job_type}")
                    job_queue.update_job(
                        job_id,
                        {"status": "error", "message": f"Unknown job type: {job_type}"},
                    )
                    job_queue.delete_job(job_id)
                    
                logger.info(f"✅ Job {job_id} finished (active jobs: {max_concurrent_jobs - job_semaphore._value}/{max_concurrent_jobs})")
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)

    while True:
        try:
            # Periodic cleanup of old files
            current_time = time.time()
            if current_time - last_cleanup_time > cleanup_interval:
                logger.info("🧹 Running periodic cleanup of old files...")
                await cleanup_old_files(max_age_hours=24)  # Clean files older than 24 hours
                last_cleanup_time = current_time
            
            # Clean up completed tasks
            completed_tasks = {t for t in active_tasks if t.done()}
            for task in completed_tasks:
                try:
                    # Get any exceptions from completed tasks
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        logger.error(f"Task exception: {exc}")
                except:
                    pass
            active_tasks -= completed_tasks
            
            # Check if we have capacity for more jobs
            if job_semaphore._value > 0:
                # Get next job from queue (non-blocking)
                job_id = job_queue.get_next_job()

                if job_id:
                    # Get job details
                    job = job_queue.get_job(job_id)
                    if not job:
                        logger.warning(f"Job {job_id} not found")
                        continue

                    job_type = job.get("type")
                    job_data_str = job.get("data", "{}")

                    # Parse job data if it's a string
                    if isinstance(job_data_str, str):
                        try:
                            job_data = json.loads(job_data_str)
                        except json.JSONDecodeError:
                            logger.error(f"Invalid job data for job {job_id}")
                            continue
                    else:
                        job_data = job_data_str

                    # Create async task for the job
                    task = asyncio.create_task(
                        process_job_with_semaphore(job_id, job_type, job_data)
                    )
                    active_tasks.add(task)
                    
                    logger.info(f"📋 Queued job {job_id} (total active: {len(active_tasks)})")
                else:
                    # No job available, sleep for a bit
                    await asyncio.sleep(2)
            else:
                # At max capacity, wait a bit before checking again
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in worker main loop: {str(e)}", exc_info=True)
            await asyncio.sleep(10)  # Wait before retrying


if __name__ == "__main__":
    asyncio.run(worker_main())
