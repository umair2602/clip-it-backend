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
# Using AssemblyAI for better speaker diarization and sentence boundaries
from services.transcription_assemblyai import transcribe_audio
from services.video_processing import generate_thumbnail, process_video, create_clip
from services.user_video_service import update_user_video, get_user_video_by_video_id, add_clip_to_video, utc_now
from utils.s3_storage import s3_client
from utils.sieve_downloader import download_youtube_video_sieve
from utils.youtube_downloader import download_youtube_video
from logging_config import setup_logging
import tempfile

# Set up logging to backend.log file
setup_logging(log_dir="logs", log_file="backend.log", log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
upload_dir = Path("uploads")
output_dir = Path("outputs")
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Note: Removed Whisper model - now using AssemblyAI cloud service


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
        
        # Release lock on error
        job_queue.release_job_lock(job_id)


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
        logger.info("📝 STEP 1: Starting transcription (AssemblyAI + Speaker Diarization)...")
        
        # AssemblyAI handles transcription in the cloud - no local model needed
        transcript = await transcribe_audio(file_path)
        
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
                        clip, idx, file_path, video_id, output_dir
                    )
            
            # Update progress - starting parallel processing
            job_queue.update_job(
                job_id,
                {
                    "progress": "75",
                    "message": f"Uploading {len(clips)} clips to S3 in parallel...",
                },
            )
            
            if original_job_id:
                job_queue.update_job(
                    original_job_id,
                    {
                        "progress": "75",
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
        
        # Release lock on error
        job_queue.release_job_lock(job_id)
        if original_job_id:
            job_queue.release_job_lock(original_job_id)


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


async def _upload_clip_to_s3_async(clip_path: str, video_id: str, clip_id: str) -> tuple:
    """Upload clip to S3 asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        file_size = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
        
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
        
        return success, s3_key
    except Exception as e:
        logger.error(f"Error uploading clip to S3: {e}")
        return False, None


async def _upload_thumbnail_to_s3_async(thumbnail_path: str, video_id: str, clip_id: str) -> tuple:
    """Upload thumbnail to S3 asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        success, s3_key = await loop.run_in_executor(
            None,
            lambda: s3_client.upload_thumbnail_to_s3(thumbnail_path, video_id, f"clip_{clip_id}_thumbnail.jpg")
        )
        return success, s3_key
    except Exception as e:
        logger.error(f"Error uploading thumbnail to S3: {e}")
        return False, None


async def process_single_clip_async(
    clip: dict,
    index: int,
    file_path: str,
    video_id: str,
    output_dir_path: Path
) -> tuple:
    """Process a single clip: create, generate thumbnail, upload to S3"""
    try:
        logger.info(f"[PARALLEL] Starting clip {index}: {clip.get('id')}")
        
        # Step 1: Create the clip file
        clips_dir = output_dir_path / video_id
        clips_dir.mkdir(exist_ok=True, parents=True)
        
        clip_filename = f"{clip.get('id')}.mp4"
        clip_path = str(clips_dir / clip_filename)
        
        # Use create_clip from video_processing service
        created_clip_path = await create_clip(
            video_path=file_path,
            output_dir=str(clips_dir),
            start_time=clip["start_time"],
            end_time=clip["end_time"],
            clip_id=clip.get("id"),
        )
        
        if not created_clip_path:
            logger.warning(f"[PARALLEL] Failed to create clip {index}")
            return index, None
        
        logger.info(f"[PARALLEL] Clip {index} created at {created_clip_path}")
        
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
            created_clip_path, thumbnail_path, thumbnail_timestamp
        )
        
        logger.info(f"[PARALLEL] Clip {index} thumbnail generated")
        
        # Step 3: Upload to S3 (clip and thumbnail in parallel)
        clip_upload = asyncio.create_task(
            _upload_clip_to_s3_async(created_clip_path, video_id, clip.get("id"))
        )
        
        thumbnail_upload = None
        if generated_path and os.path.exists(thumbnail_path) and os.path.getsize(thumbnail_path) > 0:
            thumbnail_upload = asyncio.create_task(
                _upload_thumbnail_to_s3_async(thumbnail_path, video_id, clip.get("id"))
            )
        
        # Wait for uploads
        clip_result = await clip_upload
        thumb_result = await thumbnail_upload if thumbnail_upload else (False, None)
        
        logger.info(f"[PARALLEL] Clip {index} S3 upload: clip={clip_result[0]}, thumb={thumb_result[0]}")
        
        # Step 4: Cleanup local files
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
    """Process uploaded video from S3"""
    try:
        video_id = job_data.get("video_id")
        s3_key = job_data.get("s3_key")
        user_id = job_data.get("user_id")

        logger.info(f"Processing uploaded video job {job_id} for video {video_id}")
        logger.info(f"   Job ID (task_id): {job_id}")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   S3 Key: {s3_key}")

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "downloading",
                "progress": "5",
                "message": "Downloading file from S3...",
            },
        )

        # Create temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            local_path = temp_file.name

        # Download from S3
        success = s3_client.download_from_s3(s3_key, local_path)

        if not success:
            raise Exception("Failed to download file from S3")

        # Update job status
        job_queue.update_job(
            job_id,
            {
                "status": "processing",
                "progress": "10",
                "message": "File downloaded, starting processing...",
            },
        )

        # Process the video using the same pipeline as process_video_job
        await process_video_job(job_id, {
            "video_id": video_id,
            "file_path": local_path,
            "user_id": user_id
        })
        
        # Release lock on successful completion (already handled in process_video_job)

    except Exception as e:
        logger.error(f"Error in uploaded video job {job_id}: {str(e)}", exc_info=True)
        job_queue.update_job(
            job_id,
            {
                "status": "error",
                "progress": "0",
                "message": f"Error processing uploaded video: {str(e)}",
            },
        )
        # Release lock on error
        job_queue.release_job_lock(job_id)
    finally:
        # Clean up temporary file
        try:
            if 'local_path' in locals() and os.path.exists(local_path):
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

        logger.info(f"Processing YouTube download job {job_id} for URL: {url}")
        logger.info(f"   Job ID (task_id): {job_id}")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   User ID: {user_id}")

        # Update MongoDB video status
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

        # Three-tier download fallback system
        file_path, title, video_info = None, None, None
        max_retries = 3

        # Tier 1: Try Sieve API
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Job {job_id}: [Tier 1] Attempt {attempt} to download with Sieve service")
                file_path, title, video_info = await download_youtube_video_sieve(url, video_dir)
                if file_path and title:
                    logger.info(f"Job {job_id}: ✅ Sieve download successful")
                    break
            except Exception as sieve_error:
                logger.warning(f"Job {job_id}: Sieve download failed (attempt {attempt}): {str(sieve_error)}")
                if attempt == max_retries:
                    logger.error(f"Job {job_id}: All {max_retries} Sieve attempts failed. Falling back to pytubefix.")
                    break
                await asyncio.sleep(5)

        # Tier 2: Fallback to pytubefix if Sieve failed
        if not file_path or not title:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Job {job_id}: [Tier 2] Attempt {attempt} to download with pytubefix")
                    file_path, title, video_info = await download_youtube_video(url, video_dir)
                    if file_path and title:
                        logger.info(f"Job {job_id}: ✅ pytubefix download successful")
                        break
                except Exception as pytubefix_error:
                    logger.warning(f"Job {job_id}: pytubefix download failed (attempt {attempt}): {str(pytubefix_error)}")
                    if attempt == max_retries:
                        logger.error(f"Job {job_id}: All {max_retries} pytubefix attempts failed. Falling back to yt-dlp.")
                        break
                    await asyncio.sleep(5)

        # Tier 3: Final fallback to yt-dlp
        if not file_path or not title:
            try:
                from utils.ytdlp_downloader import download_youtube_video_ytdlp
                logger.info(f"Job {job_id}: [Tier 3] Attempting download with yt-dlp (final fallback)")
                file_path, title, video_info = await download_youtube_video_ytdlp(url, video_dir)
                if file_path and title:
                    logger.info(f"Job {job_id}: ✅ yt-dlp download successful")
            except Exception as ytdlp_error:
                logger.error(f"Job {job_id}: yt-dlp download failed: {str(ytdlp_error)}")

        if not file_path or not title:
            raise Exception("Failed to download YouTube video after trying all methods (Sieve, pytubefix, yt-dlp)")

        logger.info(f"Job {job_id}: Download completed successfully: {file_path}")

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
        job_queue.update_job(
            job_id,
            {
                "status": "downloaded",
                "progress": "100",
                "message": "YouTube video downloaded successfully",
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
                },
            )
            
            # Lock already released in process_video_job

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

        job_queue.update_job(
            job_id,
            {
                "status": "error",
                "progress": "0",
                "message": f"Error: {str(e)}",
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
            else:
                # No job available, sleep for a bit
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in worker main loop: {str(e)}", exc_info=True)
            await asyncio.sleep(10)  # Wait before retrying


if __name__ == "__main__":
    asyncio.run(worker_main())
