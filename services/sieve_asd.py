"""
Sieve Active Speaker Detection Integration

This module provides cloud-based active speaker detection using Sieve's API.
Sieve uses an optimized, parallelized implementation of TalkNet + YOLO that is
~90% faster than running TalkNet locally.

Based on: https://www.sievedata.com/blog/fast-active-speaker-detection
Sieve Function: https://www.sievedata.com/functions/sieve/active_speaker_detection
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if Sieve is available
SIEVE_ASD_AVAILABLE = False

try:
    import sieve
    SIEVE_ASD_AVAILABLE = True
    logger.info("✅ Sieve SDK is available for active speaker detection")
except ImportError as e:
    logger.warning(f"Sieve SDK not available: {e}")
    SIEVE_ASD_AVAILABLE = False


async def detect_active_speaker_sieve(
    video_path: str,
    face_detections: List[Dict[str, Any]],
    fps: float
) -> Dict[int, Dict[int, float]]:
    """
    Use Sieve's cloud-based active speaker detection API.
    
    This is significantly faster than local TalkNet (~90% speedup) because:
    - Face detection (YOLO) runs in parallel with speaker detection
    - Processing is distributed across Sieve's GPU infrastructure
    - Optimized pre/post-processing
    
    Args:
        video_path: Path to video file
        face_detections: List of face detection dicts with 'frame', 'x', 'y', 'size' keys
        fps: Video frame rate
        
    Returns:
        Dict mapping frame_number -> {face_x: speaking_score}
    """
    if not SIEVE_ASD_AVAILABLE:
        logger.warning("Sieve not available, returning empty results")
        return {}
    
    try:
        logger.info(f"🎙️ Starting Sieve Active Speaker Detection for {video_path}")
        
        # Get the active speaker detection function from Sieve
        asd_function = sieve.function.get("sieve/active_speaker_detection")
        
        # Create a Sieve File object from the local video path
        video_file = sieve.File(path=video_path)
        
        # Run the active speaker detection
        # This returns a generator of results
        logger.info("   📤 Uploading video to Sieve...")
        results = asd_function.run(video_file)
        
        # Process the results
        logger.info("   ⏳ Processing with Sieve (this runs on their cloud)...")
        
        # Convert Sieve output to our expected format
        # Sieve returns: BATCHES of frames (lists of 100 frames each)
        # Each frame has: { 'frame_number': int, 'faces': [{ 'x1', 'y1', 'x2', 'y2', 'speaking_score', 'active' }] }
        frame_scores = {}
        result_count = 0
        batch_count = 0
        
        for batch in results:
            batch_count += 1
            
            # Each result is a BATCH (list of frame dicts)
            if isinstance(batch, list):
                for frame_data in batch:
                    if isinstance(frame_data, dict):
                        frame_num = frame_data.get('frame_number', 0)
                        faces = frame_data.get('faces', [])
                        
                        if frame_num not in frame_scores:
                            frame_scores[frame_num] = {}
                        
                        for face in faces:
                            if isinstance(face, dict):
                                # Get bounding box center X - Sieve uses x1, y1, x2, y2 directly
                                x1 = face.get('x1', 0)
                                x2 = face.get('x2', 0)
                                speaking_score = face.get('speaking_score', 0.0)
                                
                                if x1 and x2:
                                    center_x = int((x1 + x2) / 2)
                                    frame_scores[frame_num][center_x] = float(speaking_score)
                                    result_count += 1
            
            # Also handle the case where result is a single frame dict (for compatibility)
            elif isinstance(batch, dict):
                frame_num = batch.get('frame_number', batch.get('frame', 0))
                faces = batch.get('faces', [])
                
                if frame_num not in frame_scores:
                    frame_scores[frame_num] = {}
                
                for face in faces:
                    if isinstance(face, dict):
                        x1 = face.get('x1', 0)
                        x2 = face.get('x2', 0)
                        speaking_score = face.get('speaking_score', face.get('score', 0.0))
                        
                        if x1 and x2:
                            center_x = int((x1 + x2) / 2)
                            frame_scores[frame_num][center_x] = float(speaking_score)
                            result_count += 1
            
            # Log progress periodically
            if batch_count % 10 == 0:
                logger.debug(f"   Processed {batch_count} batches, {result_count} face detections...")
        
        logger.info(f"   ✅ Sieve ASD complete: {len(frame_scores)} frames with {result_count} face detections from {batch_count} batches")
        
        return frame_scores
        
    except Exception as e:
        logger.error(f"❌ Sieve ASD failed: {e}")
        logger.info("   Falling back to MediaPipe lip detection")
        return {}


async def detect_active_speaker_sieve_simple(
    video_path: str,
) -> List[Dict[str, Any]]:
    """
    Simplified version that just returns raw Sieve results.
    
    Useful for testing or when you want the full Sieve output format.
    
    Args:
        video_path: Path to video file
        
    Returns:
        List of detection results from Sieve
    """
    if not SIEVE_ASD_AVAILABLE:
        logger.warning("Sieve not available")
        return []
    
    try:
        asd_function = sieve.function.get("sieve/active_speaker_detection")
        video_file = sieve.File(path=video_path)
        
        results = list(asd_function.run(video_file))
        logger.info(f"Sieve returned {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.error(f"Sieve ASD failed: {e}")
        return []


def check_sieve_availability() -> bool:
    """
    Check if Sieve is properly configured and accessible.
    
    Returns:
        True if Sieve is available and API key is configured
    """
    if not SIEVE_ASD_AVAILABLE:
        return False
    
    try:
        # Check if API key is set
        api_key = os.environ.get('SIEVE_API_KEY')
        if not api_key:
            logger.warning("SIEVE_API_KEY environment variable not set")
            return False
        
        logger.info("✅ Sieve is properly configured")
        return True
        
    except Exception as e:
        logger.warning(f"Sieve availability check failed: {e}")
        return False


def _parse_sieve_results(results) -> Dict[int, Dict[int, float]]:
    """
    Parse Sieve ASD results into our expected format.
    
    Args:
        results: Generator/iterator of Sieve results (batches of frames)
        
    Returns:
        Dict mapping frame_number -> {face_x: speaking_score}
    """
    frame_scores = {}
    result_count = 0
    batch_count = 0
    
    for batch in results:
        batch_count += 1
        
        # Each result is a BATCH (list of frame dicts)
        if isinstance(batch, list):
            for frame_data in batch:
                if isinstance(frame_data, dict):
                    frame_num = frame_data.get('frame_number', 0)
                    faces = frame_data.get('faces', [])
                    
                    if frame_num not in frame_scores:
                        frame_scores[frame_num] = {}
                    
                    for face in faces:
                        if isinstance(face, dict):
                            x1 = face.get('x1', 0)
                            x2 = face.get('x2', 0)
                            speaking_score = face.get('speaking_score', 0.0)
                            
                            if x1 and x2:
                                center_x = int((x1 + x2) / 2)
                                frame_scores[frame_num][center_x] = float(speaking_score)
                                result_count += 1
        
        # Handle single frame dict (compatibility)
        elif isinstance(batch, dict):
            frame_num = batch.get('frame_number', batch.get('frame', 0))
            faces = batch.get('faces', [])
            
            if frame_num not in frame_scores:
                frame_scores[frame_num] = {}
            
            for face in faces:
                if isinstance(face, dict):
                    x1 = face.get('x1', 0)
                    x2 = face.get('x2', 0)
                    speaking_score = face.get('speaking_score', face.get('score', 0.0))
                    
                    if x1 and x2:
                        center_x = int((x1 + x2) / 2)
                        frame_scores[frame_num][center_x] = float(speaking_score)
                        result_count += 1
    
    return frame_scores, result_count, batch_count


async def batch_push_asd(
    video_paths: List[str],
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Upload multiple videos to Sieve in PARALLEL for active speaker detection.
    
    Uses .push() to submit all jobs at once, then collects results.
    This is much faster than processing one-by-one with .run().
    
    Args:
        video_paths: List of video file paths to process
        
    Returns:
        Dict mapping video_path -> {frame_number: {face_x: speaking_score}}
    """
    if not SIEVE_ASD_AVAILABLE:
        logger.warning("Sieve not available for batch processing")
        return {path: {} for path in video_paths}
    
    if not video_paths:
        return {}
    
    try:
        logger.info(f"🚀 Starting PARALLEL Sieve ASD for {len(video_paths)} videos")
        
        # Get the active speaker detection function
        asd_function = sieve.function.get("sieve/active_speaker_detection")
        
        # Submit ALL jobs in parallel using .push()
        jobs = []
        for video_path in video_paths:
            video_file = sieve.File(path=video_path)
            # .push() submits the job asynchronously and returns a future
            job = asd_function.push(video_file)
            jobs.append({
                'path': video_path,
                'future': job
            })
            logger.info(f"   📤 Submitted: {os.path.basename(video_path)}")
        
        logger.info(f"   ⏳ Waiting for {len(jobs)} Sieve jobs to complete...")
        
        # Collect results from all jobs
        results_map = {}
        for i, job_info in enumerate(jobs):
            video_path = job_info['path']
            future = job_info['future']
            
            try:
                # .result() blocks until this specific job is done
                results = future.result()
                frame_scores, result_count, batch_count = _parse_sieve_results(results)
                results_map[video_path] = frame_scores
                logger.info(f"   ✅ [{i+1}/{len(jobs)}] {os.path.basename(video_path)}: {len(frame_scores)} frames, {result_count} detections")
            except Exception as e:
                logger.error(f"   ❌ [{i+1}/{len(jobs)}] {os.path.basename(video_path)} failed: {e}")
                results_map[video_path] = {}
        
        logger.info(f"🎉 PARALLEL Sieve ASD complete: {len(results_map)} videos processed")
        return results_map
        
    except Exception as e:
        logger.error(f"❌ Batch Sieve ASD failed: {e}")
        return {path: {} for path in video_paths}


def push_asd_job(video_path: str):
    """
    Submit a single video for ASD processing without waiting for results.
    
    Use this when you want to start processing immediately and collect
    results later with get_asd_job_result().
    
    Args:
        video_path: Path to video file
        
    Returns:
        Sieve job future, or None if submission failed
    """
    if not SIEVE_ASD_AVAILABLE:
        logger.warning("Sieve not available")
        return None
    
    try:
        asd_function = sieve.function.get("sieve/active_speaker_detection")
        video_file = sieve.File(path=video_path)
        future = asd_function.push(video_file)
        logger.info(f"📤 Submitted ASD job for: {os.path.basename(video_path)}")
        return future
    except Exception as e:
        logger.error(f"Failed to submit ASD job: {e}")
        return None


def get_asd_job_result(future) -> Dict[int, Dict[int, float]]:
    """
    Get results from a previously submitted ASD job.
    
    Args:
        future: Job future from push_asd_job()
        
    Returns:
        Dict mapping frame_number -> {face_x: speaking_score}
    """
    if future is None:
        return {}
    
    try:
        results = future.result()
        frame_scores, _, _ = _parse_sieve_results(results)
        return frame_scores
    except Exception as e:
        logger.error(f"Failed to get ASD job result: {e}")
        return {}


# Check availability on module load
if SIEVE_ASD_AVAILABLE:
    check_sieve_availability()

