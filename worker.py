import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from jobs import job_queue
from services.content_analysis import analyze_content

# Import services
from services.transcription import load_model, transcribe_audio
from services.video_processing import generate_thumbnail, process_video
from utils.s3_storage import s3_client
from utils.sieve_downloader import download_youtube_video_sieve
from utils.youtube_downloader import download_youtube_video
from services.user_video_service import update_user_video, get_user_video_by_video_id, utc_now
from logging_config import setup_logging

# Set up logging to backend.log file
setup_logging(log_dir="logs", log_file="backend.log", log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
upload_dir = Path("uploads")
output_dir = Path("outputs")
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Global Whisper model
whisper_model = None


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


async def initialize_worker():
    """Initialize the worker with the Whisper model"""
    global whisper_model
    try:
        logger.info("Initializing worker - loading Whisper model...")
        # Check for GPU availability
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU detected. Using CPU for transcription")

        whisper_model = load_model(model_size="tiny")
        logger.info("Worker initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing worker: {str(e)}", exc_info=True)


async def process_youtube_download_job(job_id: str, job_data: dict):
    """Process YouTube download job"""
    try:
        url = job_data.get("url")
        video_id = job_data.get("video_id")
        auto_process = job_data.get("auto_process", True)

        logger.info(f"Processing YouTube download job {job_id} for URL: {url}")

        # Update MongoDB video status
        user_id = job_data.get("user_id")
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "downloading",
                "updated_at": utc_now()
            })

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "downloading",
                "progress": "10",
                "message": "Downloading video from YouTube...",
            },
        )

        # Create video directory
        video_dir = upload_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Download the video with retry mechanism
        file_path, title, video_info = None, None, None
        max_retries = 15
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Job {job_id}: Attempt {attempt} to download with Sieve service")
                file_path, title, video_info = await download_youtube_video_sieve(
                    url, video_dir
                )
                if file_path and title:
                    break  # Success
            except Exception as sieve_error:
                logger.warning(f"Job {job_id}: Sieve download failed (attempt {attempt}): {str(sieve_error)}")
                if attempt == max_retries:
                    logger.error(f"Job {job_id}: All {max_retries} attempts failed.")
                    break
                await asyncio.sleep(5)  # Wait before retrying
        # If still not successful, try direct downloader as fallback (with retries)
        if not file_path or not title:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Job {job_id}: Attempt {attempt} to download with direct downloader as fallback")
                    file_path, title, video_info = await download_youtube_video(
                        url, video_dir
                    )
                    if file_path and title:
                        break  # Success
                except Exception as fallback_error:
                    logger.warning(f"Job {job_id}: Direct download failed (attempt {attempt}): {str(fallback_error)}")
                    if attempt == max_retries:
                        logger.error(f"Job {job_id}: All {max_retries} fallback attempts failed.")
                        break
                    await asyncio.sleep(5)  # Wait before retrying

        if not file_path or not title:
            raise Exception(
                f"Failed to download YouTube video after {max_retries} attempts with both Sieve and direct methods"
            )

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
        job_queue.update_job(
            job_id,
            {
                "status": "downloaded",
                "progress": "100",
                "message": "YouTube video downloaded successfully",
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

            # CRITICAL FIX: Update MongoDB with process_task_id so frontend can find it
            # This prevents 404 errors when frontend switches to polling the process task
            # Status stays "downloaded" since download is complete and processing is about to start
            logger.info(f"🔍 DEBUG: About to update process_task_id")
            logger.info(f"   user_id: {user_id}")
            logger.info(f"   video_id: {video_id}")
            logger.info(f"   process_job_id: {process_job_id}")
            
            if user_id:
                update_result = await update_user_video(user_id, video_id, {
                    "process_task_id": process_job_id,
                    "updated_at": utc_now()
                })
                logger.info(f"✅ Update result: {update_result}")
                logger.info(f"✅ Updated MongoDB: video {video_id} with process_task_id: {process_job_id}")
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
        
        job_queue.update_job(
            job_id,
            {
                "status": "error",
                "progress": "0",
                "message": f"Error processing YouTube download: {str(e)}",
            },
        )


async def process_video_job(job_id: str, job_data: dict):
    """Process video processing job"""
    try:
        video_id = job_data.get("video_id")
        file_path = job_data.get("file_path")
        original_job_id = job_data.get("original_job_id")
        user_id = job_data.get("user_id")

        # If user_id is not provided, fetch it from DB
        if not user_id:
            video_info = await get_user_video_by_video_id(video_id)
            if video_info:
                user_id = video_info.get("user_id")
            else:
                logger.error(f"Could not find user for video {video_id}")
                user_id = None

        logger.info(f"Processing video job {job_id} for video {video_id}")
        
        # ⏱️ START PIPELINE TIMING
        import time
        pipeline_start = time.time()
        logger.info("="*70)
        logger.info("🚀 STARTING VIDEO PROCESSING PIPELINE")
        logger.info("="*70)

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "transcribing",
                "updated_at": utc_now()
            })

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "transcribing",
                "progress": "30",
                "message": "Transcribing audio",
            },
        )

        # Also update original job if it exists
        if original_job_id:
            job_queue.update_job(
                original_job_id,
                {
                    "status": "transcribing",
                    "progress": "30",
                    "message": "Transcribing audio",
                },
            )

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at path: {file_path}")

        # ⏱️ STEP 1: TRANSCRIPTION
        step_start = time.time()
        logger.info("📝 STEP 1: Starting transcription (Whisper + Diarization + Alignment)...")
        
        transcript = await transcribe_audio(
            file_path, 
            model_size="base",
            job_id=job_id,
            job_queue=job_queue
        )
        if not transcript:
            raise ValueError("Transcription returned None")
        
        step_elapsed = time.time() - step_start
        logger.info(f"✅ STEP 1 COMPLETE: Transcription finished in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
        logger.info(f"   - Total segments: {len(transcript.get('segments', []))}")
        logger.info(f"   - Has speaker labels: {'Yes' if transcript.get('transcript_for_ai') else 'No'}")

        # Handle empty segments gracefully - this is valid for silent/corrupted videos
        segments_count = len(transcript.get("segments", []))
        logger.info(f"Transcription completed with {segments_count} segments")

        if segments_count == 0:
            logger.warning("Video appears to be silent or have no speech content")
            # Continue processing with empty transcript - don't fail

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "analyzing",
                "updated_at": utc_now()
            })

        # Update job status
        job_queue.update_job(
            job_id, {"status": "analyzing", "progress": "50", "message": "Analyzing content"},
        )

        if original_job_id:
            job_queue.update_job(
                original_job_id,
                {
                    "status": "analyzing",
                    "progress": "50",
                    "message": "Analyzing content",
                },
            )

        # ⏱️ STEP 2: AI CONTENT ANALYSIS
        step_start = time.time()
        logger.info("🤖 STEP 2: Starting AI content analysis (OpenAI clip detection)...")
        
        segments = await analyze_content(transcript)
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

        # Update MongoDB video status
        if user_id:
            await update_user_video(user_id, video_id, {
                "status": "processing",
                "updated_at": utc_now()
            })

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "processing",
                "progress": "70",
                "message": "Processing video clips",
            },
        )

        if original_job_id:
            job_queue.update_job(
                original_job_id,
                {
                    "status": "processing",
                    "progress": "70",
                    "message": "Processing video clips",
                },
            )

        # Create output directory
        clips_dir = output_dir / video_id
        clips_dir.mkdir(exist_ok=True)

        # ⏱️ STEP 3: VIDEO PROCESSING
        step_start = time.time()
        logger.info("🎬 STEP 3: Starting video clip creation (FFmpeg processing)...")
        
        clips = await process_video(file_path, transcript, segments)
        
        step_elapsed = time.time() - step_start
        logger.info(f"✅ STEP 3 COMPLETE: Video processing finished in {step_elapsed:.2f} seconds")
        logger.info(f"   - Clips created: {len(clips) if clips else 0}")

        # Generate thumbnails for each clip
        for i, clip in enumerate(clips):
            progress = 60 + (i * 30 / max(len(clips), 1))

            job_queue.update_job(
                job_id,
                {
                    "progress": str(progress),
                    "message": f"Processing clip {i + 1}/{len(clips)}",
                },
            )

            if original_job_id:
                job_queue.update_job(
                    original_job_id,
                    {
                        "progress": str(progress),
                        "message": f"Processing clip {i + 1}/{len(clips)}",
                    },
                )

            # Generate thumbnail
            clip_path = clip.get("path")
            if clip_path and os.path.exists(clip_path):
                thumbnail_filename = f"{clip.get('id')}_thumbnail.jpg"
                thumbnail_path = os.path.abspath(str(clips_dir / thumbnail_filename))

                clip_duration = clip.get("end_time", 0) - clip.get("start_time", 0)
                thumbnail_timestamp = (
                    min(clip_duration / 3, clip_duration - 0.5)
                    if clip_duration > 1.5
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
                        clips[i]["thumbnail_path"] = thumbnail_path
                        clips[i]["thumbnail_url"] = (
                            f"/outputs/{video_id}/{thumbnail_filename}"
                        )
                    else:
                        clips[i]["thumbnail_url"] = "/static/default_thumbnail.jpg"
                except Exception as thumb_error:
                    logger.error(f"Error generating thumbnail: {str(thumb_error)}")
                    clips[i]["thumbnail_url"] = "/static/default_thumbnail.jpg"
            else:
                clips[i]["thumbnail_url"] = "/static/default_thumbnail.jpg"

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
        job_queue.update_job(
            job_id,
            {
                "status": "completed",
                "progress": "100",
                "message": "Processing completed",
                "clips": json.dumps(clips),
            },
        )

        if original_job_id:
            job_queue.update_job(
                original_job_id,
                {
                    "status": "completed",
                    "progress": "100",
                    "message": "Processing completed",
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

        # Clean up uploaded video file and directory after successful processing
        try:
            video_upload_dir = upload_dir / video_id
            if video_upload_dir.exists():
                shutil.rmtree(video_upload_dir)
                logger.info(f"✅ Cleaned up uploaded video directory: {video_upload_dir}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️  Failed to clean up uploaded video directory {video_upload_dir}: {cleanup_error}")

        logger.info(f"Job {job_id} completed successfully with {len(clips)} clips")

    except Exception as e:
        logger.error(f"Error in video processing job {job_id}: {str(e)}", exc_info=True)
        job_queue.update_job(
            job_id, {"status": "error", "progress": "0", "message": f"Error: {str(e)}"}
        )

        if original_job_id:
            job_queue.update_job(
                original_job_id,
                {"status": "error", "progress": "0", "message": f"Error: {str(e)}"},
            )
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


async def process_manual_clip_job(job_id: str, job_data: dict):
    """Process manual clip generation job"""
    try:
        video_id = job_data.get("video_id")
        video_path = job_data.get("video_path")
        start_time = float(job_data.get("start_time"))
        end_time = float(job_data.get("end_time"))
        title = job_data.get("title", "Manual Clip")

        logger.info(f"Processing manual clip job {job_id} for video {video_id}")

        # Update job status
        job_queue.update_job(
            job_id,
            {"status": "processing", "progress": "20", "message": "Processing clip"},
        )

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
        job_queue.update_job(
            job_id,
            {
                "status": "completed",
                "progress": "100",
                "message": "Clip generated successfully",
                "clip": json.dumps(clip),
            },
        )

        logger.info(f"Manual clip job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error in manual clip job {job_id}: {str(e)}", exc_info=True)
        job_queue.update_job(
            job_id, {"status": "error", "progress": "0", "message": f"Error: {str(e)}"}
        )


async def process_s3_download_job(job_id: str, job_data: dict):
    """Process S3 download and process job"""
    try:
        video_id = job_data.get("video_id")
        s3_key = job_data.get("s3_key")
        local_path = job_data.get("local_path")

        logger.info(f"Processing S3 download job {job_id} for video {video_id}")

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "downloading",
                "progress": "5",
                "message": "Downloading file from S3...",
            },
        )

        # Download from S3
        success = s3_client.download_from_s3(s3_key, local_path)

        if not success:
            raise Exception("Failed to download file from S3")

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "downloaded",
                "progress": "10",
                "message": "File downloaded from S3, starting processing",
            },
        )

        # Now process the video like a regular processing job
        await process_video_job(job_id, {"video_id": video_id, "file_path": local_path})

    except Exception as e:
        logger.error(f"Error in S3 download job {job_id}: {str(e)}", exc_info=True)
        job_queue.update_job(
            job_id,
            {
                "status": "error",
                "progress": "0",
                "message": f"Error downloading from S3: {str(e)}",
            },
        )


async def worker_main():
    """Main worker loop"""
    logger.info("Starting worker...")

    # Initialize worker
    await initialize_worker()

    logger.info("Worker ready, waiting for jobs...")
    
    # Track last cleanup time
    import time
    last_cleanup_time = time.time()
    cleanup_interval = 3600  # Run cleanup every hour

    while True:
        try:
            # Periodic cleanup of old files
            current_time = time.time()
            if current_time - last_cleanup_time > cleanup_interval:
                logger.info("🧹 Running periodic cleanup of old files...")
                await cleanup_old_files(max_age_hours=24)  # Clean files older than 24 hours
                last_cleanup_time = current_time
            
            # Get next job from queue
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

                logger.info(f"Processing job {job_id} of type {job_type}")

                # Process job based on type
                if job_type == "youtube_download":
                    await process_youtube_download_job(job_id, job_data)
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
            else:
                # No job available, sleep for a bit
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in worker main loop: {str(e)}", exc_info=True)
            await asyncio.sleep(10)  # Wait before retrying


if __name__ == "__main__":
    asyncio.run(worker_main())
