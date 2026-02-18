import json
import logging
import os
import subprocess
# Import configuration
import sys
import tempfile
import time
import asyncio
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set up logging first (before imports that might use it)
logger = logging.getLogger(__name__)

# ===== CANCELLATION CONTEXT SYSTEM =====
# Context variables to pass user_id and video_id to nested functions for cancellation checks
_cancellation_user_id: ContextVar[Optional[str]] = ContextVar('cancellation_user_id', default=None)
_cancellation_video_id: ContextVar[Optional[str]] = ContextVar('cancellation_video_id', default=None)

class ProcessingCancelledException(Exception):
    """Exception raised when processing is cancelled by user"""
    pass

def set_cancellation_context(user_id: str, video_id: str):
    """Set the cancellation context for nested function calls"""
    _cancellation_user_id.set(user_id)
    _cancellation_video_id.set(video_id)
    logger.info(f"🔒 Cancellation context set: user={user_id}, video={video_id}")

def clear_cancellation_context():
    """Clear the cancellation context"""
    _cancellation_user_id.set(None)
    _cancellation_video_id.set(None)

async def check_cancellation():
    """Check if processing has been cancelled. Call this periodically in long-running operations.
    
    Raises:
        ProcessingCancelledException: If the video processing has been cancelled
    """
    user_id = _cancellation_user_id.get()
    video_id = _cancellation_video_id.get()
    
    if not user_id or not video_id:
        return  # No context set, can't check
    
    try:
        from services.user_video_service import get_user_video
        video = await get_user_video(user_id, video_id)
        
        if video and video.status == "failed":
            error_msg = video.error_message or "Unknown reason"
            if "cancelled" in error_msg.lower():
                logger.warning(f"🛑 Processing cancelled by user for video {video_id}")
                raise ProcessingCancelledException(f"Processing cancelled: {error_msg}")
    except ProcessingCancelledException:
        raise
    except Exception as e:
        # Don't fail the entire process if we can't check cancellation
        logger.debug(f"Could not check cancellation status: {e}")

def check_cancellation_sync():
    """Synchronous version of check_cancellation for use in sync code.
    Creates a new event loop if needed.
    
    Raises:
        ProcessingCancelledException: If the video processing has been cancelled
    """
    user_id = _cancellation_user_id.get()
    video_id = _cancellation_video_id.get()
    
    if not user_id or not video_id:
        return  # No context set, can't check
    
    try:
        # Try to get the running loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't run another coroutine synchronously
            # Schedule it as a task instead - but this won't block
            # For sync code in async context, we just skip the check
            return
        except RuntimeError:
            # No running loop, create one
            pass
        
        # Run the async check in a new loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_check_cancellation_async(user_id, video_id))
        finally:
            loop.close()
    except ProcessingCancelledException:
        raise
    except Exception as e:
        logger.debug(f"Could not check cancellation status (sync): {e}")

async def _check_cancellation_async(user_id: str, video_id: str):
    """Internal async cancellation check"""
    from services.user_video_service import get_user_video
    video = await get_user_video(user_id, video_id)
    
    if video and video.status == "failed":
        error_msg = video.error_message or "Unknown reason"
        if "cancelled" in error_msg.lower():
            raise ProcessingCancelledException(f"Processing cancelled: {error_msg}")
# ===== END CANCELLATION CONTEXT SYSTEM =====

# Import other services
# face tracking removed to simplify and speed up clip creation

# Constants for speaker tracking and camera movement
MIN_SPEAKER_DURATION = 2.0  # Minimum seconds a speaker must talk before camera switches (increased for stability)
SPEAKER_HYSTERESIS = 250  # Pixels of "stickiness" - how much to favor current speaker position (increased to reduce jitter)
FRAME_SKIP_INTERVAL = 5  # Process every Nth frame for speaker detection (reduces sensitivity to quick movements)
SMOOTHING_WINDOW = 3  # Number of frames to average for speaker position smoothing


# TalkNet Active Speaker Detection - high accuracy audio-visual model
try:
    from services.talknet_asd import (
        TalkNetASD,
        detect_active_speaker_simple,
        check_talknet_installation,
        TALKNET_AVAILABLE
    )
    if TALKNET_AVAILABLE:
        logger.info("✅ TalkNet ASD is available - using high-accuracy speaker detection")
    else:
        logger.info("ℹ️ TalkNet ASD not yet initialized - will download model on first use")
except ImportError as e:
    logger.warning(f"TalkNet ASD not available, will use basic visual lip movement detection: {e}")
    TALKNET_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

async def process_video(
    video_path: str, transcript: Dict[str, Any], segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process video to create clips for each engaging segment (IN PARALLEL).

    Args:
        video_path: Path to the video file
        transcript: Transcript from AssemblyAI transcription service
        segments: List of engaging segments with start and end times

    Returns:
        List of processed clips with paths
    """
    try:
        logger.info("="*70)
        logger.info(f"📹 STARTING VIDEO CLIP CREATION (OPTIMIZED MODE)")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Number of segments to process: {len(segments)}")
        logger.info("="*70)
        
        # 🚀 OPTIMIZATION: Map speakers ONCE from first 10 seconds (instead of analyzing every frame)
        # This reduces CPU usage by 90%+ and prevents auto-scaling issues
        from services.speaker_mapping import map_speakers_from_sample
        
        logger.info("\n" + "="*70)
        logger.info("🎯 STEP 1: INTELLIGENT SPEAKER MAPPING (ONE-TIME ANALYSIS)")
        logger.info("   Running TalkNet ONLY on first 10 seconds to identify speakers...")
        logger.info("   This replaces frame-by-frame analysis and saves 90%+ compute time")
        logger.info("="*70)
        
        global_speaker_map = await map_speakers_from_sample(
            video_path=video_path,
            transcript=transcript,
            sample_duration=10.0  # Only analyze first 10 seconds
        )
        
        # Track discovered speakers across clips (mutable dict for learning new speakers)
        discovered_speakers = global_speaker_map.copy()
        
        logger.info(f"\n✅ Speaker mapping complete! Found {len(discovered_speakers)} speakers:")
        for speaker, x_pos in discovered_speakers.items():
            logger.info(f"   Speaker {speaker}: X position = {x_pos}")
        logger.info("="*70 + "\n")
        
        # Create temporary output directory for processing
        import tempfile
        import asyncio
        temp_dir = tempfile.mkdtemp(prefix="clip_processing_")
        output_dir = Path(temp_dir)
        logger.info(f"Created temporary output directory: {output_dir}")

        logger.info("\n" + "="*70)
        logger.info("SEGMENTS TO PROCESS:")
        for i, seg in enumerate(segments):
            duration = seg.get('end_time', 0) - seg.get('start_time', 0)
            logger.info(f"  Segment {i+1}: {seg.get('title', 'Untitled')} ({seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s, duration: {duration:.1f}s)")
        logger.info("="*70 + "\n")

        # Process clips in PARALLEL
        total_segments = len(segments)
        MAX_CONCURRENT_CLIPS = 3  # Process 3 clips at once
        
        logger.info(f"🚀 Processing {total_segments} clips with {MAX_CONCURRENT_CLIPS} concurrent workers")
        
        async def process_single_clip(i: int, segment: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single clip (used for parallel execution)"""
            # Check for cancellation before starting each clip
            await check_cancellation()
            
            segment_start_time = time.time()
            clip_id = f"clip_{i}"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎬 Processing Clip {i+1}/{total_segments}: {segment.get('title', 'Untitled')}")
            logger.info(f"   Timestamp: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
            logger.info(f"{'='*70}")

            # Create the clip
            logger.info(f"   ⏳ Step 1/3: Creating video clip...")
            clip_result = await create_clip(
                video_path=video_path,
                output_dir=str(output_dir),
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                clip_id=clip_id,
                transcript=transcript,
                speaker_map=discovered_speakers,  # Pass mutable speaker map
            )
            
            # Unpack result - may include new speaker discoveries
            if isinstance(clip_result, tuple):
                clip_path, new_speakers = clip_result
                # Merge newly discovered speakers into global map
                if new_speakers:
                    logger.info(f"   🆕 SPEAKER DISCOVERY: Found {len(new_speakers)} new speaker(s) in this clip")
                    for speaker, pos in new_speakers.items():
                        if speaker not in discovered_speakers:
                            discovered_speakers[speaker] = pos
                            logger.info(f"   ✅ CACHED NEW SPEAKER: {speaker} → X={pos} (will use for future clips)")
                        else:
                            logger.debug(f"   ℹ️  Speaker {speaker} already cached at X={discovered_speakers[speaker]}")
            else:
                clip_path = clip_result

            # Only process if clip was created
            if clip_path is not None:
                logger.info(f"   ✅ Clip created successfully: {clip_path}")
                
                # Generate thumbnail for this clip
                logger.info(f"   ⏳ Step 2/3: Generating thumbnail...")
                thumbnail_filename = f"{clip_id}_thumbnail.jpg"
                thumbnail_path = str(output_dir / thumbnail_filename)

                try:
                    # Generate the thumbnail
                    await generate_thumbnail(
                        video_path=clip_path, output_path=thumbnail_path
                    )
                    logger.info(f"   ✅ Thumbnail generated: {thumbnail_path}")

                    # Verify the thumbnail exists
                    if not os.path.exists(thumbnail_path):
                        logger.warning(
                            f"   ⚠️  Thumbnail file does not exist at {thumbnail_path} despite successful generation"
                        )
                        thumbnail_path = None
                except Exception as thumb_error:
                    logger.error(f"   ❌ Failed to generate thumbnail: {str(thumb_error)}")
                    thumbnail_path = None

                # Construct the thumbnail URL properly (will be set to S3 URL later)
                thumbnail_url = None
                if thumbnail_path and os.path.exists(thumbnail_path):
                    logger.info(f"   ✅ Thumbnail ready at: {thumbnail_path}")
                else:
                    logger.warning(
                        f"   ⚠️  Thumbnail doesn't exist at {thumbnail_path}, not setting URL"
                    )

                segment_elapsed = time.time() - segment_start_time
                logger.info(f"   ⏱️  Clip {i+1}/{total_segments} completed in {segment_elapsed:.2f}s")
                logger.info(f"   ✅ Successfully processed: {segment.get('title', 'Untitled')}")
                
                return {
                    "id": clip_id,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "title": segment.get("title", f"Clip {i + 1}"),
                    "description": segment.get("description", ""),
                    "path": clip_path,
                    "url": None,  # Will be set to S3 URL later
                    "thumbnail_path": thumbnail_path,
                    "thumbnail_url": thumbnail_url,
                }
            else:
                logger.warning(f"   ❌ Clip {i+1}/{total_segments} SKIPPED - No faces detected or creation failed")
                logger.warning(f"   Segment: {segment.get('title', 'Untitled')}")
                return None
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLIPS)
        
        async def process_with_limit(i: int, segment: Dict[str, Any]) -> Dict[str, Any]:
            """Process clip with concurrency limit"""
            async with semaphore:
                return await process_single_clip(i, segment)
        
        # Create tasks for all clips
        tasks = [process_with_limit(i, segment) for i, segment in enumerate(segments)]
        
        # Run all tasks in parallel
        parallel_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parallel_elapsed = time.time() - parallel_start
        
        # Filter out None results and exceptions
        processed_clips = []
        for result in results:
            if result and not isinstance(result, Exception):
                processed_clips.append(result)
            elif isinstance(result, Exception):
                logger.error(f"❌ Clip processing raised exception: {result}")
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ VIDEO PROCESSING COMPLETE (PARALLEL)")
        logger.info(f"Total clips created: {len(processed_clips)} out of {total_segments} segments")
        logger.info(f"⏱️  Parallel processing time: {parallel_elapsed:.2f}s")
        if len(segments) > 1:
            sequential_estimate = parallel_elapsed * len(segments) / MAX_CONCURRENT_CLIPS
            speedup = sequential_estimate / parallel_elapsed if parallel_elapsed > 0 else 0
            logger.info(f"📈 Estimated speedup vs sequential: ~{speedup:.1f}x")
        
        # Log speaker discovery summary
        total_speakers = len(discovered_speakers)
        initial_speakers = len(global_speaker_map) if global_speaker_map else 0
        new_speakers_found = total_speakers - initial_speakers
        
        if new_speakers_found > 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"🎯 SPEAKER DISCOVERY SUMMARY")
            logger.info(f"   Initial mapped speakers: {initial_speakers} {list(global_speaker_map.keys()) if global_speaker_map else []}")
            logger.info(f"   Newly discovered speakers: {new_speakers_found}")
            logger.info(f"   Total speakers in video: {total_speakers}")
            logger.info(f"   Final speaker map: {discovered_speakers}")
            logger.info(f"{'='*70}")
        
        logger.info("="*70 + "\n")

        return processed_clips

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        raise


async def create_clip(
    video_path: str, 
    output_dir: str, 
    start_time: float, 
    end_time: float, 
    clip_id: str, 
    transcript: Dict[str, Any] = None,
    speaker_map: Dict[str, int] = None  # Pre-computed speaker positions
) -> tuple:  # Returns (clip_path, newly_discovered_speakers)
    """
    Extract video segment and convert to vertical (9:16) format using OPTIMIZED speaker tracking.
    
    OPTIMIZATION: Instead of running TalkNet on every frame:
    1. Uses pre-computed speaker_map (from first 10s analysis)
    2. Generates crop positions from transcript timestamps
    3. If unmapped speaker found, runs TalkNet and caches new speaker for future clips
    4. Reduces CPU usage by 70-90%+ and prevents auto-scaling issues
    
    Returns:
        Tuple of (clip_path, newly_discovered_speakers_dict)
    """
    logger.info(f"🎬 Creating clip {clip_id}: {start_time:.2f}s - {end_time:.2f}s")
    
    # Check for cancellation at start
    await check_cancellation()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")
    temp_path = os.path.join(output_dir, f"{clip_id}_temp.mp4")
    
    # Step 1: Extract the segment
    logger.info(f"   📂 Extracting segment...")
    extract_clip(video_path, temp_path, start_time, end_time)
    
    # Check for cancellation after extraction
    await check_cancellation()
    
    # Step 2: Generate crop positions from transcript (OPTIMIZED - no heavy computation!)
    newly_discovered_speakers = {}  # Track any new speakers found
    
    if speaker_map:
        logger.info(f"   🚀 Using optimized transcript-based cropping (90%+ faster than TalkNet!)...")
        crop_positions = await generate_optimized_crop_positions(
            temp_path, start_time, end_time, transcript, speaker_map
        )
        
        # If empty list returned, it means clip has unmapped speakers - fall back and learn
        if not crop_positions:
            # Log which speakers are in the clip vs which are known
            clip_speakers = set()
            if transcript and 'utterances' in transcript:
                for utt in transcript['utterances']:
                    if float(utt.get('start', 0)) < end_time and float(utt.get('end', 0)) > start_time:
                        clip_speakers.add(utt.get('speaker', 'Unknown'))
            missing_speakers = clip_speakers - set(speaker_map.keys())
            logger.warning(f"   ⚠️ UNMAPPED SPEAKERS DETECTED: {missing_speakers}")
            logger.info(f"   🔍 Known speakers: {set(speaker_map.keys())}")
            logger.info(f"   🎯 Running TalkNet on this clip to discover missing speakers...")
            
            crop_positions, new_speakers = await detect_talknet_crop_positions_with_discovery(
                temp_path, start_time, end_time, transcript, speaker_map
            )
            newly_discovered_speakers = new_speakers
    else:
        # Fallback to old method if no speaker map
        logger.warning(f"   ⚠️ No speaker map available, falling back to full TalkNet analysis...")
        crop_positions = await detect_talknet_crop_positions(temp_path, start_time, end_time, transcript)
    
    # Check for cancellation after TalkNet analysis
    await check_cancellation()
    
    # Step 3: Apply smart crop with smooth transitions
    logger.info(f"   🎥 Applying TalkNet-guided speaker tracking...")
    await apply_smart_crop_with_transitions(temp_path, output_path, crop_positions)
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    logger.info(f"   ✅ Clip created with TalkNet speaker tracking: {output_path}")
    return output_path, newly_discovered_speakers


async def generate_optimized_crop_positions(
    video_path: str,
    start_time: float,
    end_time: float,
    transcript: Dict[str, Any],
    speaker_map: Dict[str, int]
) -> list:
    """
    OPTIMIZED: Generate crop positions from transcript timestamps and speaker map.
    
    This is 90%+ faster than frame-by-frame TalkNet analysis!
    Uses the speaker map created from the first 10 seconds to intelligently crop.
    """
    import cv2
    from services.speaker_mapping import generate_crop_positions_from_transcript
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Calculate crop dimensions for 9:16 aspect ratio (vertical video)
    crop_h = input_h
    crop_w = int(crop_h * 0.5625)  # 9:16 aspect ratio to fill vertical space
    if crop_w > input_w:
        crop_w = input_w
        crop_h = int(crop_w / 0.5625)  # Adjust height if width is constrained
    
    logger.info(f"      ✂️  Crop dimensions: {crop_w}x{crop_h} (9:16 aspect ratio)")
    logger.info(f"      📊 Using {len(speaker_map)} pre-mapped speakers")
    
    # Generate positions directly from transcript (NO heavy computation!)
    crop_position_tuples = generate_crop_positions_from_transcript(
        transcript=transcript,
        speaker_map=speaker_map,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
        input_w=input_w,
        crop_w=crop_w
    )
    
    # Convert to old format for compatibility (crop_x, center_x, etc.)
    crop_positions = []
    for frame_num, center_x in crop_position_tuples:
        # Calculate crop_x from center_x
        crop_x = int(center_x - crop_w / 2)
        crop_x = max(0, min(crop_x, input_w - crop_w))  # Bounds check
        
        crop_positions.append({
            'frame': frame_num,
            'crop_x': crop_x,
            'center_x': center_x,
            'has_detection': True
        })
    
    logger.info(f"      ✅ Generated {len(crop_positions)} crop positions (instant - no TalkNet needed!)")
    
    return crop_positions


async def detect_talknet_crop_positions_with_discovery(
    video_path: str,
    start_time: float,
    end_time: float,
    transcript: Dict[str, Any] = None,
    speaker_map: Dict[str, float] = None
) -> tuple:
    """
    Run TalkNet ASD to detect crop positions AND discover new speakers for caching.
    
    This is used when an unmapped speaker is encountered (e.g., Speaker C appears at minute 5).
    Instead of just generating crop positions, we also extract and return the speaker's X position
    so it can be cached for future clips.
    
    Returns:
        tuple: (crop_positions list, newly_discovered_speakers dict)
               e.g., ([{...crop data...}], {'C': 800})
    """
    logger.info(f"      🔍 Running TalkNet with speaker discovery mode...")
    
    # Run full TalkNet analysis
    crop_positions = await detect_talknet_crop_positions(video_path, start_time, end_time, transcript)
    
    # Extract speaker positions from the crop results
    newly_discovered_speakers = {}
    
    if transcript and 'utterances' in transcript and speaker_map is not None:
        # Get all speakers in this clip
        clip_speakers = set()
        for utterance in transcript['utterances']:
            utt_start = float(utterance.get('start', 0))
            utt_end = float(utterance.get('end', 0))
            speaker = utterance.get('speaker', 'Unknown')
            
            # Check if utterance overlaps with clip
            if utt_start < end_time and utt_end > start_time:
                clip_speakers.add(speaker)
        
        # Find speakers that aren't in the map yet
        unmapped_speakers = {s for s in clip_speakers if s not in speaker_map}
        
        if unmapped_speakers:
            logger.info(f"      🆕 Found {len(unmapped_speakers)} unmapped speaker(s): {unmapped_speakers}")
            
            # For each unmapped speaker, extract their X position from crop_positions
            for speaker in unmapped_speakers:
                # Get all utterances for this speaker
                speaker_utterances = [
                    utt for utt in transcript['utterances']
                    if utt.get('speaker') == speaker and
                    float(utt.get('start', 0)) < end_time and
                    float(utt.get('end', 0)) > start_time
                ]
                
                if speaker_utterances:
                    # Collect X positions from crop_positions during this speaker's utterances
                    speaker_x_positions = []
                    
                    for crop in crop_positions:
                        crop_time = crop.get('time', 0)
                        
                        # Check if this crop falls within any of this speaker's utterances
                        for utt in speaker_utterances:
                            utt_start = float(utt.get('start', 0))
                            utt_end = float(utt.get('end', 0))
                            
                            if utt_start <= crop_time <= utt_end:
                                # Use center_x if available, otherwise crop_x
                                x_pos = crop.get('center_x') or crop.get('crop_x')
                                if x_pos is not None:
                                    speaker_x_positions.append(x_pos)
                                break
                    
                    if speaker_x_positions:
                        # Use median X position for stability
                        median_x = int(np.median(speaker_x_positions))
                        newly_discovered_speakers[speaker] = median_x
                        logger.info(f"      ✅ SPEAKER EXTRACTED: {speaker} → X={median_x}")
                        logger.info(f"         📊 Analyzed {len(speaker_x_positions)} frame samples during {len(speaker_utterances)} utterance(s)")
                        logger.info(f"         📍 Position range: X={min(speaker_x_positions)}-{max(speaker_x_positions)} (median={median_x})")
                    else:
                        logger.warning(f"      ⚠️ EXTRACTION FAILED: Could not determine X position for speaker {speaker}")
                        logger.warning(f"         Checked {len(speaker_utterances)} utterance(s) but found no matching crop positions")
    
    return crop_positions, newly_discovered_speakers


async def detect_talknet_crop_positions(
    video_path: str,
    start_time: float, 
    end_time: float,
    transcript: Dict[str, Any] = None
) -> list:
    """
    Use TalkNet ASD (Active Speaker Detection) to detect faces and identify who is speaking.
    TalkNet uses audio-visual cross-attention for accurate speaker detection.
    
    Strategy:
    1. Detect all faces in each frame using basic face detection
    2. Use TalkNet to determine which face is actively speaking
    3. Track the active speaker's position across frames
    4. Generate smooth crop positions that follow the speaker
    5. Fallback to visual cues (lip movement) only if TalkNet fails
    """
    import cv2
    import mediapipe as mp
    import numpy as np
    
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_detection
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"      Video: {input_w}x{input_h} @ {fps:.2f}fps, {total_frames} frames")
    
    # Calculate crop dimensions for 9:16 aspect ratio (vertical video)
    crop_h = input_h
    crop_w = int(crop_h * 0.5625)  # 9:16 aspect ratio to fill vertical space
    
    # Ensure crop width doesn't exceed video width
    if crop_w > input_w:
        crop_w = input_w
        crop_h = int(crop_w / 0.5625)  # Adjust height if width is constrained
    
    logger.info(f"      ✂️  Crop dimensions: {crop_w}x{crop_h} (9:16 aspect ratio)")
    
    # STEP 1: Build speaker timeline from transcript
    logger.info(f"      🔍 Checking transcript data...")
    logger.info(f"         Transcript type: {type(transcript)}")
    logger.info(f"         Transcript keys: {transcript.keys() if transcript and isinstance(transcript, dict) else 'None'}")
    
    speaker_timeline = build_speaker_timeline(transcript, start_time, end_time)
    if speaker_timeline:
        logger.info(f"      📢 Speaker timeline: {len(speaker_timeline)} utterances")
        for utt in speaker_timeline[:3]:  # Log first 3
            logger.info(f"         {utt['speaker']}: {utt['start']:.1f}s - {utt['end']:.1f}s")
    else:
        logger.warning(f"      ⚠️ No speaker timeline data found - will use visual tracking only")
        logger.warning(f"         This means crop won't follow active speaker intelligently")
    
    # STEP 2: Detect all faces and track basic lip movement
    # NOTE: MediaPipe Face Mesh is used ONLY for face detection and basic lip movement tracking
    # TalkNet ASD (below) handles the actual speaker detection using audio-visual cross-attention
    person_detections = []  # List of all detected people with lip movement data
    # FRAME SKIPPING: Reduce sampling rate to stabilize speaker selection
    # Lower FPS = less sensitivity to momentary speaker changes = smoother tracking
    sample_interval = max(1, int(fps / FRAME_SKIP_INTERVAL))  # Process every Nth frame based on FPS
    
    mp_face_mesh = mp.solutions.face_mesh
    
    # Lip landmark indices for basic visual tracking (fallback only)
    # Upper lip: 61, 291, 0  |  Lower lip: 17, 314, 0
    # We'll track vertical distance between upper and lower lips
    UPPER_LIP_INDICES = [13, 14]  # Top lip landmarks
    LOWER_LIP_INDICES = [78, 308]  # Bottom lip landmarks
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,  # Track up to 5 people
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        frame_idx = 0
        prev_lip_distances = {}  # Track lip distance per person for movement detection
        cancellation_check_counter = 0  # Counter for periodic cancellation checks
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Periodic cancellation check (every 30 sampled frames, roughly every 3 seconds)
            cancellation_check_counter += 1
            if cancellation_check_counter >= 30:
                cancellation_check_counter = 0
                await check_cancellation()
            
            # Sample frames for efficiency
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue
            
            # Current timestamp in the clip
            frame_time = (frame_idx / fps)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and lips
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Process each detected face
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Get face bounding box for position
                    x_coords = [landmark.x for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y for landmark in face_landmarks.landmark]
                    
                    face_center_x = int(np.mean(x_coords) * input_w)
                    face_center_y = int(np.mean(y_coords) * input_h)
                    
                    face_width = (max(x_coords) - min(x_coords)) * input_w
                    face_height = (max(y_coords) - min(y_coords)) * input_h
                    face_size = (face_width * face_height) / (input_w * input_h)  # Normalized size
                    
                    # Calculate lip distance (vertical opening)
                    upper_lip_y = np.mean([face_landmarks.landmark[i].y for i in UPPER_LIP_INDICES])
                    lower_lip_y = np.mean([face_landmarks.landmark[i].y for i in LOWER_LIP_INDICES])
                    lip_distance = abs(lower_lip_y - upper_lip_y) * input_h
                    
                    # Calculate lip movement (change from previous frame)
                    person_key = f"person_{face_center_x // 100}"  # Group by rough X position
                    lip_movement = 0
                    
                    if person_key in prev_lip_distances:
                        lip_movement = abs(lip_distance - prev_lip_distances[person_key])
                    
                    prev_lip_distances[person_key] = lip_distance
                    
                    # IMPROVED: Adaptive lip movement threshold based on face size and resolution
                    # Larger faces (closer to camera) have larger absolute lip movements
                    # Normalize threshold: base_threshold * sqrt(face_size) * resolution_factor
                    base_threshold = 1.5  # Base threshold in pixels
                    resolution_factor = input_h / 1080.0  # Normalize for different resolutions
                    size_factor = max(0.5, min(2.0, np.sqrt(face_size) * 10))  # Scale by face size
                    adaptive_threshold = base_threshold * size_factor * resolution_factor
                    
                    is_speaking = lip_movement > adaptive_threshold
                    
                    # IMPROVED: Calculate confidence score based on face size, position, and lip movement
                    # Confidence components:
                    # 1. Face size (larger = more confident, 0-1 scale)
                    size_confidence = min(1.0, face_size * 20)  # Normalize face size to 0-1
                    # 2. Lip movement strength (how much above threshold, 0-1 scale)
                    movement_strength = min(1.0, lip_movement / (adaptive_threshold * 3)) if is_speaking else 0.0
                    # 3. Face position (center faces are more likely to be main speaker, 0-1 scale)
                    center_distance = abs(face_center_x - input_w / 2) / (input_w / 2)
                    position_confidence = 1.0 - (center_distance * 0.5)  # Center gets 1.0, edges get 0.5
                    
                    # Combined confidence (weighted average)
                    detection_confidence = (
                        size_confidence * 0.3 +  # 30% weight on size
                        movement_strength * 0.5 +  # 50% weight on lip movement
                        position_confidence * 0.2  # 20% weight on position
                    )
                    
                    person_detections.append({
                        'frame': frame_idx,
                        'time': frame_time,
                        'x': face_center_x,
                        'y': face_center_y,
                        'size': face_size,
                        'lip_distance': lip_distance,
                        'lip_movement': lip_movement,
                        'is_speaking': is_speaking,
                        'confidence': detection_confidence,
                        'adaptive_threshold': adaptive_threshold,  # Store for debugging
                        'face_landmarks': face_landmarks,  # Store for potential future use
                        'type': 'face_mesh'
                    })
            
            frame_idx += 1
        
        cap.release()
    
    logger.info(f"      ✅ Detected {len(person_detections)} person instances across frames")
    
    if len(person_detections) == 0:
        logger.warning(f"      ⚠️ No people detected - using center crop")
        return [{'frame': 0, 'crop_x': (input_w - crop_w) // 2, 'center_x': input_w // 2, 'has_detection': False}]
    
    # STEP 2.5: TALKNET ENHANCEMENT - Use TalkNet for accurate speaking detection
    # TalkNet uses audio-visual cross-attention to determine who is ACTUALLY speaking
    try:
        if TALKNET_AVAILABLE:
            logger.info(f"      🎙️ Running TalkNet ASD for accurate speaker detection...")
            talknet_scores = await detect_active_speaker_simple(video_path, person_detections, fps)
            
            if talknet_scores:
                # Update detections with TalkNet scores
                talknet_updates = 0
                for det in person_detections:
                    frame = det['frame']
                    x = det['x']
                    
                    if frame in talknet_scores:
                        # Find the closest match by X position
                        frame_scores = talknet_scores[frame]
                        best_score = 0.0
                        for face_x, score in frame_scores.items():
                            if abs(face_x - x) < 150:  # Within 150 pixels
                                best_score = max(best_score, score)
                        
                        if best_score > 0:
                            # Override the is_speaking flag with TalkNet's score
                            det['is_speaking'] = best_score > 0.5
                            det['talknet_score'] = best_score
                            det['confidence'] = (det['confidence'] + best_score) / 2  # Blend confidence
                            talknet_updates += 1
                
                logger.info(f"      ✅ TalkNet updated {talknet_updates} detections with high-accuracy speaking scores")
            else:
                logger.info(f"      ℹ️ TalkNet returned no scores, using basic visual lip movement detection")
        else:
            logger.info(f"      ℹ️ TalkNet not available, using basic visual lip movement detection")
    except Exception as e:
        logger.warning(f"      ⚠️ TalkNet failed, using basic visual lip movement detection: {e}")
    
    # STEP 3: Cluster people by spatial position to identify distinct individuals
    person_clusters = cluster_people_by_position(person_detections, input_w)
    logger.info(f"      👥 Identified {len(person_clusters)} distinct people in video")
    
    # Log speaker timeline info (for debugging, but we won't use speaker labels for mapping)
    if speaker_timeline:
        logger.info(f"      📢 Speaker timeline has {len(speaker_timeline)} utterances (using for timing only, not speaker identity)")
    
    # STEP 4: SIMPLIFIED - Follow visual activity when audio is present
    # Instead of trying to map "Speaker A" to a position, we just find WHO is visually active
    logger.info(f"      🎯 Using VISUAL-ACTIVITY tracking (ignoring speaker labels)")
    
    # STEP 5: Generate crop positions based on visual activity at each moment
    crop_positions = []
    
    # Get all unique frame numbers we sampled
    sampled_frames = sorted(set(d['frame'] for d in person_detections))
    
    # Track current position for hysteresis (smooth transitions)
    current_center_x = None
    last_switch_time = 0
    
    # ACTIVE SPEAKER TRACKING: Remember who we're currently following
    active_speaker_x = None  # X position of current active speaker
    active_speaker_tolerance = 300  # Pixels tolerance to consider "same speaker"
    
    # Check if audio is playing at any given time using speaker timeline
    def is_audio_active(time: float) -> bool:
        """Check if someone is speaking at this time based on transcript"""
        for utt in speaker_timeline:
            if utt['start'] <= time <= utt['end']:
                return True
        return False
    
    for frame_num in sampled_frames:
        frame_time = frame_num / fps
        
        # Get all detections for this frame
        frame_detections = [d for d in person_detections if d['frame'] == frame_num]
        
        if not frame_detections:
            # No detections - use last known position or center
            center_x = current_center_x if current_center_x is not None else input_w // 2
        else:
            # Check if audio is active (someone is speaking)
            audio_active = is_audio_active(frame_time) if speaker_timeline else True
            
            if audio_active:
                # AUDIO IS PLAYING - find the person with most visual activity (lip movement)
                # If TalkNet is available, use talknet_score; otherwise use is_speaking flag
                speaking_detections = [d for d in frame_detections if d.get('is_speaking', False)]
                
                if speaking_detections:
                    # CRITICAL FIX: If we're already tracking a speaker, STRONGLY prefer them
                    # This prevents switching to a nearby speaker and centering between them
                    if active_speaker_x is not None:
                        # Find if our current speaker is still speaking
                        current_speaker_still_speaking = [
                            d for d in speaking_detections 
                            if abs(d['x'] - active_speaker_x) < active_speaker_tolerance
                        ]
                        
                        if current_speaker_still_speaking:
                            # Current speaker is still active - STAY with them
                            best_speaker = max(current_speaker_still_speaking, key=lambda d: (
                                d.get('talknet_score', 0) * 100 +
                                d.get('lip_movement', 0) * 10 +
                                d.get('confidence', 0.5) * 5 +
                                d.get('size', 0) * 100
                            ))
                            target_x = best_speaker['x']
                            logger.debug(f"      Frame {frame_num}: Continuing with active speaker at X={target_x}")
                        else:
                            # Current speaker stopped - find new speaker
                            best_speaker = max(speaking_detections, key=lambda d: (
                                d.get('talknet_score', 0) * 100 +
                                d.get('lip_movement', 0) * 10 +
                                d.get('confidence', 0.5) * 5 +
                                d.get('size', 0) * 100
                            ))
                            target_x = best_speaker['x']
                            active_speaker_x = target_x  # Update active speaker
                            logger.debug(f"      Frame {frame_num}: New speaker detected at X={target_x}")
                    else:
                        # No active speaker yet - pick the best one
                        best_speaker = max(speaking_detections, key=lambda d: (
                            d.get('talknet_score', 0) * 100 +
                            d.get('lip_movement', 0) * 10 +
                            d.get('confidence', 0.5) * 5 +
                            d.get('size', 0) * 100
                        ))
                        target_x = best_speaker['x']
                        active_speaker_x = target_x  # Set initial active speaker
                        logger.debug(f"      Frame {frame_num}: Initial speaker at X={target_x}")
                    
                    talknet_info = f", talknet={best_speaker.get('talknet_score', 'N/A'):.2f}" if 'talknet_score' in best_speaker else ""
                    logger.debug(f"         Details: lip_movement={best_speaker.get('lip_movement', 0):.2f}{talknet_info}")
                else:
                    # No one visually speaking but audio is active
                    # If we have an active speaker, stay with them
                    if active_speaker_x is not None:
                        # Find person closest to active speaker position
                        closest = min(frame_detections, key=lambda d: abs(d['x'] - active_speaker_x))
                        if abs(closest['x'] - active_speaker_x) < active_speaker_tolerance:
                            target_x = closest['x']
                            logger.debug(f"      Frame {frame_num}: Staying with active speaker (no visual speech) at X={target_x}")
                        else:
                            # Active speaker moved too far, pick largest
                            largest = max(frame_detections, key=lambda d: d.get('size', 0))
                            target_x = largest['x']
                            active_speaker_x = target_x
                            logger.debug(f"      Frame {frame_num}: Active speaker lost, following largest at X={target_x}")
                    else:
                        # Fall back to largest/most prominent person (likely the speaker)
                        largest = max(frame_detections, key=lambda d: d.get('size', 0))
                        target_x = largest['x']
                        active_speaker_x = target_x
                        logger.debug(f"      Frame {frame_num}: Audio active, following largest person at X={target_x}")
            else:
                # NO AUDIO - stay with active speaker if we have one, else follow largest
                if active_speaker_x is not None:
                    # Find person closest to active speaker
                    closest = min(frame_detections, key=lambda d: abs(d['x'] - active_speaker_x))
                    if abs(closest['x'] - active_speaker_x) < active_speaker_tolerance:
                        target_x = closest['x']
                    else:
                        largest = max(frame_detections, key=lambda d: d.get('size', 0))
                        target_x = largest['x']
                        active_speaker_x = target_x
                else:
                    # No active speaker - follow the largest/most prominent person
                    largest = max(frame_detections, key=lambda d: d.get('size', 0))
                    target_x = largest['x']
                    active_speaker_x = target_x
                logger.debug(f"      Frame {frame_num} ({frame_time:.1f}s): No audio, following person at X={target_x}")
            
            # CRITICAL FIX: Don't smooth positions - use target_x directly to avoid centering between speakers
            # Smoothing was causing the camera to average positions when switching speakers
            # Instead, rely on hysteresis and MIN_SPEAKER_DURATION for stability
            
            # Apply hysteresis to prevent jittery switching
            if current_center_x is not None:
                distance = abs(target_x - current_center_x)  # Use target_x directly, not smoothed
                time_since_switch = frame_time - last_switch_time
                
                # Only switch if:
                # 1. Distance is significant (> hysteresis threshold) AND
                # 2. Enough time has passed since last switch
                if distance < SPEAKER_HYSTERESIS or time_since_switch < MIN_SPEAKER_DURATION:
                    center_x = current_center_x  # Stay with current position
                    logger.debug(f"      Frame {frame_num}: Hysteresis applied - staying at X={current_center_x} (target={target_x}, distance={distance}px, time_since_switch={time_since_switch:.1f}s)")
                else:
                    center_x = target_x  # Switch to new speaker directly
                    current_center_x = center_x
                    last_switch_time = frame_time
                    logger.debug(f"      Frame {frame_num}: Speaker switch to X={center_x} (distance={distance}px, time_since_switch={time_since_switch:.1f}s)")
            else:
                # First frame - use target position directly
                center_x = target_x
                current_center_x = center_x
                last_switch_time = frame_time
        
        # Calculate crop position
        crop_x = center_x - (crop_w // 2)
        
        # Add safety margins - if person is near edge, ensure they're not cut off
        min_safe_margin = 100  # pixels from edge
        
        if center_x < crop_w // 2 + min_safe_margin:
            crop_x = 0
        elif center_x > input_w - crop_w // 2 - min_safe_margin:
            crop_x = input_w - crop_w
        
        # Final bounds check
        crop_x = max(0, min(crop_x, input_w - crop_w))
        
        # Calculate frame time relative to clip start
        frame_time = start_time + (frame_num / fps)
        
        crop_positions.append({
            'frame': frame_num,
            'time': frame_time,  # Add time for speaker discovery
            'crop_x': crop_x,
            'center_x': center_x,
            'has_detection': len(frame_detections) > 0
        })
    
    if len(crop_positions) == 0:
        logger.warning(f"      ⚠️ No crop positions generated - using center")
        return [{'frame': 0, 'time': start_time, 'crop_x': (input_w - crop_w) // 2, 'center_x': input_w // 2, 'has_detection': False}]
    
    # Log summary
    unique_positions = len(set(p['crop_x'] for p in crop_positions))
    logger.info(f"      ✅ Generated {len(crop_positions)} crop positions ({unique_positions} unique X positions)")
    
    return crop_positions


def build_speaker_timeline(transcript: Dict[str, Any], start_time: float, end_time: float) -> list:
    """
    Extract speaker utterances from transcript that overlap with the clip segment.
    Returns list of {speaker, start, end} dicts in clip-relative time.
    
    Handles both AssemblyAI format (utterances) and custom format (speakers + segments).
    """
    if not transcript:
        logger.debug("      No transcript provided")
        return []
    
    timeline = []
    
    # FORMAT 1: AssemblyAI format with 'utterances' key
    if 'utterances' in transcript:
        logger.debug(f"      Using AssemblyAI utterances format ({len(transcript['utterances'])} utterances)")
        for utterance in transcript['utterances']:
            # Times are already in seconds (converted in transcription_assemblyai.py)
            utt_start = float(utterance.get('start', 0))
            utt_end = float(utterance.get('end', 0))
            speaker = utterance.get('speaker', 'Unknown')
            
            # Check if utterance overlaps with our clip segment
            if utt_start < end_time and utt_end > start_time:
                # Convert to clip-relative time (0-based)
                clip_start = max(0, utt_start - start_time)
                clip_end = min(end_time - start_time, utt_end - start_time)
                
                timeline.append({
                    'speaker': speaker,
                    'start': clip_start,
                    'end': clip_end
                })
    
    # FORMAT 2: Custom format with 'speakers' array
    elif 'speakers' in transcript and isinstance(transcript['speakers'], list):
        logger.debug(f"      Using custom speakers format ({len(transcript['speakers'])} speaker entries)")
        for speaker_entry in transcript['speakers']:
            # speaker_entry might be: {speaker: "A", start: 10.5, end: 15.2, text: "..."}
            if isinstance(speaker_entry, dict):
                spk_start = float(speaker_entry.get('start', 0))
                spk_end = float(speaker_entry.get('end', 0))
                speaker = speaker_entry.get('speaker', 'Unknown')
                
                # Check if speaker segment overlaps with our clip
                if spk_start < end_time and spk_end > start_time:
                    clip_start = max(0, spk_start - start_time)
                    clip_end = min(end_time - start_time, spk_end - start_time)
                    
                    timeline.append({
                        'speaker': speaker,
                        'start': clip_start,
                        'end': clip_end
                    })
    
    # FORMAT 3: Try segments with speaker info
    elif 'segments' in transcript:
        logger.debug(f"      Checking segments format ({len(transcript['segments'])} segments)")
        for segment in transcript['segments']:
            if isinstance(segment, dict) and 'speaker' in segment:
                seg_start = float(segment.get('start', 0))
                seg_end = float(segment.get('end', 0))
                speaker = segment.get('speaker', 'Unknown')
                
                if seg_start < end_time and seg_end > start_time:
                    clip_start = max(0, seg_start - start_time)
                    clip_end = min(end_time - start_time, seg_end - start_time)
                    
                    timeline.append({
                        'speaker': speaker,
                        'start': clip_start,
                        'end': clip_end
                    })
    
    else:
        logger.debug(f"      No speaker data found in transcript. Available keys: {list(transcript.keys())}")
    
    return timeline




def cluster_people_by_position(detections: list, frame_width: int) -> dict:
    """
    Cluster detected people by their X position to identify distinct individuals.
    Returns dict: {cluster_id: {avg_x, avg_y, count}}
    """
    import numpy as np
    
    if not detections:
        return {}
    
    # Extract X positions
    x_positions = [d['x'] for d in detections]
    
    # Simple clustering: group people within 200px of each other
    CLUSTER_THRESHOLD = 200
    clusters = {}
    cluster_id = 0
    
    for detection in detections:
        x = detection['x']
        y = detection['y']
        
        # Find if this belongs to existing cluster
        matched_cluster = None
        for cid, cluster in clusters.items():
            if abs(x - cluster['avg_x']) < CLUSTER_THRESHOLD:
                matched_cluster = cid
                break
        
        if matched_cluster is not None:
            # Add to existing cluster
            cluster = clusters[matched_cluster]
            cluster['x_positions'].append(x)
            cluster['y_positions'].append(y)
            cluster['count'] += 1
            cluster['avg_x'] = int(np.mean(cluster['x_positions']))
            cluster['avg_y'] = int(np.mean(cluster['y_positions']))
        else:
            # Create new cluster
            clusters[cluster_id] = {
                'avg_x': x,
                'avg_y': y,
                'x_positions': [x],
                'y_positions': [y],
                'count': 1
            }
            cluster_id += 1
    
    return clusters



async def apply_smart_crop_with_transitions(temp_path: str, output_path: str, crop_positions: list):
    """
    IMPROVED: Apply dynamic crop to video with smooth transitions that follow active speakers.
    Uses Python/OpenCV for frame-by-frame processing to achieve true dynamic cropping.
    """
    import cv2
    
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate crop dimensions for 9:16 aspect ratio (vertical video)
    crop_h = input_h
    crop_w = int(crop_h * 0.5625)  # 9:16 aspect ratio to fill vertical space
    
    # Ensure crop width doesn't exceed video width
    if crop_w > input_w:
        crop_w = input_w
        crop_h = int(crop_w / 0.5625)  # Adjust height if width is constrained
    
    logger.info(f"      📐 Video: {input_w}x{input_h}, {total_frames} frames @ {fps:.1f}fps")
    logger.info(f"      ✂️  Crop dimensions: {crop_w}x{crop_h} (9:16 aspect ratio)")
    
    # IMPROVED: Build frame-by-frame crop positions with smooth interpolation
    if len(crop_positions) > 1 and crop_w < input_w:
        # Only do dynamic cropping if we're actually cropping (not using full width)
        import numpy as np
        from bisect import bisect_left, bisect_right
        
        # Create keyframe lookup (sorted by frame number)
        # This is memory efficient - only stores keyframes, not every frame
        keyframe_frames = []
        keyframe_crop_x = []
        
        for pos in crop_positions:
            keyframe_frames.append(pos['frame'])
            keyframe_crop_x.append(pos['crop_x'])
        
        # Log keyframe positions
        logger.info(f"      🎯 Crop keyframes ({len(crop_positions)} positions):")
        for i, pos in enumerate(crop_positions[:5]):  # Log first 5
            time_point = pos['frame'] / fps
            speaker = pos.get('active_speaker', 'unknown')
            logger.info(f"         Frame {pos['frame']} ({time_point:.1f}s): X={pos['crop_x']}, Speaker={speaker}")
        if len(crop_positions) > 5:
            logger.info(f"         ... and {len(crop_positions) - 5} more keyframes")
        
        def get_crop_x_for_frame(frame_idx: int) -> int:
            """Get interpolated crop_x for a specific frame using binary search. O(log n) lookup."""
            if not keyframe_frames:
                return (input_w - crop_w) // 2  # Center fallback
            
            # Binary search to find surrounding keyframes
            idx = bisect_right(keyframe_frames, frame_idx)
            
            if idx == 0:
                # Before first keyframe - use first keyframe's value
                crop_x = keyframe_crop_x[0]
            elif idx >= len(keyframe_frames):
                # After last keyframe - use last keyframe's value
                crop_x = keyframe_crop_x[-1]
            else:
                # Between two keyframes - interpolate
                f1 = keyframe_frames[idx - 1]
                f2 = keyframe_frames[idx]
                x1 = keyframe_crop_x[idx - 1]
                x2 = keyframe_crop_x[idx]
                
                # Linear interpolation with smoothstep
                if f2 != f1:
                    alpha = (frame_idx - f1) / (f2 - f1)
                    alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)  # Smoothstep
                    crop_x = int(x1 + (x2 - x1) * alpha_smooth)
                else:
                    crop_x = x1
            
            # Clamp to valid range
            return max(0, min(crop_x, input_w - crop_w))
        
        # Log crop movement statistics using keyframes
        if keyframe_crop_x:
            crop_range = max(keyframe_crop_x) - min(keyframe_crop_x)
            logger.info(f"      📊 Crop movement: min={min(keyframe_crop_x):.0f}, max={max(keyframe_crop_x):.0f}, range={crop_range:.0f}px")
        
        # Use OpenCV for dynamic frame-by-frame cropping with smooth speaker following
        logger.info(f"      🎬 Applying dynamic crop with OpenCV frame-by-frame processing...")
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get video codec info
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_path + '.temp.mp4'
        
        # Create video writer for cropped frames
        out = cv2.VideoWriter(temp_output, fourcc, fps, (crop_w, crop_h))
        
        if not out.isOpened():
            logger.error(f"      ❌ Failed to open VideoWriter, falling back to static crop")
            cap.release()
            # Fallback to static crop - use most common keyframe position
            from collections import Counter
            dominant_crop_x = Counter(keyframe_crop_x).most_common(1)[0][0] if keyframe_crop_x else (input_w - crop_w) // 2
            filter_str = f"crop={crop_w}:{crop_h}:{int(dominant_crop_x)}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            cmd = ["ffmpeg", "-y", "-i", temp_path, "-vf", filter_str, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return
        
        frame_idx = 0
        last_log_time = 0
        frames_written = 0
        errors_count = 0
        last_good_crop_x = get_crop_x_for_frame(0)
        
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get crop position using O(log n) binary search lookup
                crop_x = get_crop_x_for_frame(frame_idx)
                last_good_crop_x = crop_x  # Save for recovery
                
                # Crop the frame
                cropped_frame = frame[0:crop_h, crop_x:crop_x+crop_w]
                
                # Verify frame dimensions before writing
                if cropped_frame.shape[0] == crop_h and cropped_frame.shape[1] == crop_w:
                    out.write(cropped_frame)
                    frames_written += 1
                else:
                    # If dimensions don't match, create a black frame
                    logger.warning(f"         Frame {frame_idx}: Invalid crop dimensions, using black frame")
                    black_frame = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                    out.write(black_frame)
                    frames_written += 1
                
            except Exception as frame_error:
                errors_count += 1
                if errors_count <= 5:  # Only log first 5 errors
                    logger.warning(f"         Frame {frame_idx} error: {frame_error}")
                # Write a frame with last known good position to maintain continuity
                try:
                    if 'frame' in dir() and frame is not None:
                        safe_crop_x = max(0, min(int(last_good_crop_x), input_w - crop_w))
                        safe_frame = frame[0:crop_h, safe_crop_x:safe_crop_x+crop_w]
                        out.write(safe_frame)
                        frames_written += 1
                except:
                    pass  # Skip this frame entirely if we can't recover
            
            # Log progress every 2 seconds
            current_time = frame_idx / fps
            if current_time - last_log_time >= 2.0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"         Progress: {progress:.1f}% (frame {frame_idx}/{total_frames}, crop_x={crop_x})")
                last_log_time = current_time
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        logger.info(f"      📊 Processed {frame_idx} frames, wrote {frames_written}, errors: {errors_count}")
        
        # Check if we got a valid output
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
            logger.info(f"      ✅ Dynamic cropping complete, re-encoding with GPU (NVENC)...")
            
            # Re-encode with h264 and audio using ffmpeg - try GPU first, fallback to CPU
            hw_accel_cmd = get_hardware_acceleration_cmd()
            use_gpu = hw_accel_cmd and "cuda" in " ".join(hw_accel_cmd)
            
            if use_gpu:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hwaccel", "cuda",
                    "-i", temp_output,
                    "-i", temp_path,  # Original for audio
                    "-map", "0:v",  # Video from temp_output
                    "-map", "1:a?",  # Audio from original (if exists)
                    "-vf", "scale=1080:1920",  # Scale to exact 9:16 dimensions (no padding needed)
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",  # NVENC preset (p1=fastest, p7=quality)
                    "-rc", "constqp",
                    "-qp", "23",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    output_path
                ]
                logger.info(f"      🚀 Using NVENC GPU encoding")
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", temp_output,
                    "-i", temp_path,  # Original for audio
                    "-map", "0:v",  # Video from temp_output
                    "-map", "1:a?",  # Audio from original (if exists)
                    "-vf", "scale=1080:1920",  # Scale to exact 9:16 dimensions (no padding needed)
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    output_path
                ]
                logger.info(f"      💻 Using CPU (libx264) encoding - no GPU detected")
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as nvenc_error:
                if use_gpu:
                    logger.warning(f"      ⚠️ NVENC encoding failed, falling back to CPU: {nvenc_error}")
                    cmd = [
                        "ffmpeg", "-y", "-i", temp_output, "-i", temp_path,
                        "-map", "0:v", "-map", "1:a?",
                        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                else:
                    raise
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            logger.info(f"      ✅ Dynamic speaker-following crop completed successfully")
        else:
            # Fallback to static crop if OpenCV output failed
            logger.warning(f"      ⚠️ OpenCV output empty, falling back to static crop")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            from collections import Counter
            dominant_crop_x = Counter(keyframe_crop_x).most_common(1)[0][0] if keyframe_crop_x else (input_w - crop_w) // 2
            filter_str = f"crop={crop_w}:{crop_h}:{int(dominant_crop_x)}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
            hw_fb = get_hardware_acceleration_cmd()
            if hw_fb and "cuda" in " ".join(hw_fb):
                cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-i", temp_path, "-vf", filter_str, "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", temp_path, "-vf", filter_str, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # Fallback to CPU if GPU fails
                cmd = ["ffmpeg", "-y", "-i", temp_path, "-vf", filter_str, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path]
                subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"      ✅ Static fallback crop completed")
            
    else:
        # Single position - use static crop with ffmpeg (faster)
        crop_x = crop_positions[0]['crop_x']
        logger.info(f"      📍 Using static crop at X={crop_x}")
        filter_str = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        
        # Use GPU encoding if available
        hw_single = get_hardware_acceleration_cmd()
        use_gpu_single = hw_single and "cuda" in " ".join(hw_single)
        
        if use_gpu_single:
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "cuda",
                "-i", temp_path, "-vf", filter_str,
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
            ]
            logger.info(f"      🚀 Crop filter (GPU): {filter_str}")
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_path, "-vf", filter_str,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
            ]
            logger.info(f"      💻 Crop filter (CPU): {filter_str}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as single_error:
            if use_gpu_single:
                logger.warning(f"      ⚠️ GPU encoding failed, falling back to CPU: {single_error}")
                cmd = [
                    "ffmpeg", "-y", "-i", temp_path, "-vf", filter_str,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
                ]
                subprocess.run(cmd, check=True)
            else:
                raise
        logger.info(f"      ✅ Crop applied successfully")
    
    # Check if we're using full width (no cropping needed)
    if crop_w >= input_w:
        # Full width mode - no cropping, just scale and pad to 9:16
        logger.info(f"      📍 Using full frame (no crop) with padding to 9:16")
        
        # Calculate padding needed for 9:16 aspect ratio (1080:1920 = 0.5625)
        target_aspect = 9 / 16  # 0.5625
        current_aspect = input_w / input_h
        
        if current_aspect > target_aspect:
            # Video is wider than 9:16 - add vertical padding (top/bottom black bars)
            scale_filter = f"scale=1080:-1,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        else:
            # Video is taller than 9:16 - add horizontal padding (left/right black bars)
            scale_filter = f"scale=-1:1920,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        
        # Use GPU encoding if available for full-width processing
        hw_full = get_hardware_acceleration_cmd()
        use_gpu_full = hw_full and "cuda" in " ".join(hw_full)
        
        if use_gpu_full:
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "cuda",
                "-i", temp_path, "-vf", scale_filter,
                "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
            ]
            logger.info(f"      🚀 Scale + pad filter (GPU): {scale_filter}")
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_path, "-vf", scale_filter,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
            ]
            logger.info(f"      💻 Scale + pad filter (CPU): {scale_filter}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as full_error:
            if use_gpu_full:
                logger.warning(f"      ⚠️ GPU encoding failed for full-frame, falling back to CPU: {full_error}")
                cmd = [
                    "ffmpeg", "-y", "-i", temp_path, "-vf", scale_filter,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path
                ]
                subprocess.run(cmd, check=True)
            else:
                raise
        logger.info(f"      ✅ Full frame processed with padding")
# async def create_clip(
#     video_path: str, output_dir: str, start_time: float, end_time: float, clip_id: str
# ) -> str:
#     """Create a social media ready clip from the video.

#     Args:
#         video_path: Path to the video file
#         output_dir: Directory to save the output
#         start_time: Start time of the clip in seconds
#         end_time: End time of the clip in seconds
#         clip_id: Unique identifier for the clip

#     Returns:
#         Path to the processed clip or None if no suitable clip could be created
#     """
#     try:
#         # Create temporary directory for processing
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # 1. First analyze the original segment to find continuous face segments
#             # Extract a temporary clip for face analysis
#             temp_analysis_path = os.path.join(temp_dir, f"{clip_id}_analysis.mp4")
#             extract_clip(video_path, temp_analysis_path, start_time, end_time)

#             # Track faces in the analysis clip
#             try:
#                 face_tracking_data = await track_faces(temp_analysis_path)
#                 print(
#                     f"Face tracking completed: {len(face_tracking_data)} frames processed"
#                 )

#                 # Check if any faces were detected in any frames
#                 has_faces = any(frame.get("faces") for frame in face_tracking_data)
#                 if not has_faces:
#                     print(
#                         f"No faces detected in clip {clip_id}. Skipping this clip entirely."
#                     )
#                     # Return None to indicate this clip should be skipped
#                     return None

#                 # Find continuous segments with faces
#                 face_segments = find_continuous_face_segments(face_tracking_data)
#                 if not face_segments:
#                     print(
#                         f"No continuous face segments found in clip {clip_id}. Skipping this clip."
#                     )
#                     return None

#                 # Use the longest continuous segment with faces
#                 longest_segment = max(
#                     face_segments, key=lambda x: x["end_frame"] - x["start_frame"]
#                 )
#                 print(
#                     f"Using longest face segment: frames {longest_segment['start_frame']} to {longest_segment['end_frame']}"
#                 )

#                 # Calculate the actual start and end times for the face segment
#                 fps = len(face_tracking_data) / (end_time - start_time)
#                 face_segment_start = start_time + (longest_segment["start_frame"] / fps)
#                 face_segment_end = start_time + (longest_segment["end_frame"] / fps)

#                 # Ensure the segment is at least 1 second long
#                 if face_segment_end - face_segment_start < 1.0:
#                     # Extend the clip if it's too short
#                     extension_needed = 1.0 - (face_segment_end - face_segment_start)
#                     face_segment_end += extension_needed / 2
#                     face_segment_start -= extension_needed / 2
#                     # Ensure we don't go outside the original clip boundaries
#                     face_segment_start = max(start_time, face_segment_start)
#                     face_segment_end = min(end_time, face_segment_end)

#                 print(
#                     f"Extracting face segment from {face_segment_start:.2f}s to {face_segment_end:.2f}s"
#                 )

#                 # Extract the clip with faces
#                 raw_clip_path = os.path.join(temp_dir, f"{clip_id}_raw.mp4")
#                 extract_clip(
#                     video_path, raw_clip_path, face_segment_start, face_segment_end
#                 )

#                 # Re-track faces in the extracted clip to get accurate face positions
#                 face_tracking_data = await track_faces(raw_clip_path)

#             except Exception as face_error:
#                 print(
#                     f"Face tracking failed: {str(face_error)}. Using fallback method."
#                 )
#                 # Extract the original clip as fallback
#                 raw_clip_path = os.path.join(temp_dir, f"{clip_id}_raw.mp4")
#                 extract_clip(video_path, raw_clip_path, start_time, end_time)
#                 face_tracking_data = []  # Empty list as fallback

#             # 3. Convert to vertical format with face tracking (or center crop if tracking failed)
#             vertical_clip_path = os.path.join(temp_dir, f"{clip_id}_vertical.mp4")
#             try:
#                 convert_to_vertical(
#                     raw_clip_path, vertical_clip_path, face_tracking_data
#                 )
#             except Exception as convert_error:
#                 print(
#                     f"Vertical conversion failed: {str(convert_error)}. Using simple crop."
#                 )
#                 # Fallback to a simpler conversion without face tracking
#                 convert_to_vertical(raw_clip_path, vertical_clip_path, [])

#             # 4. Add captions and game overlay
#             final_clip_path = os.path.join(output_dir, f"{clip_id}.mp4")
#             add_captions_and_game(
#                 vertical_clip_path, final_clip_path, start_time, end_time
#             )

#             return final_clip_path

#     except Exception as e:
#         print(f"Error creating clip: {str(e)}")
#         raise


# face-detection helpers removed — pipeline simplified to center crop/letterbox


def extract_clip(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    """Extract a clip from the video.

    Args:
        input_path: Path to the input video
        output_path: Path to save the output clip
        start_time: Start time of the clip in seconds
        end_time: End time of the clip in seconds
    """
    try:
        # Calculate duration
        duration = end_time - start_time

        # Use hardware acceleration if available
        hw_accel_cmd = get_hardware_acceleration_cmd()

        if hw_accel_cmd and "cuda" in " ".join(hw_accel_cmd):
            try:
                # First try with basic CUDA acceleration but with less intensive settings
                # This approach puts the decoder on CPU and encoder on GPU to avoid memory issues
                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-ss",
                    str(start_time),  # FAST SEEK: Place -ss before -i
                    "-i",
                    input_path,
                    "-t",
                    str(duration),
                    "-c:v",
                    "h264_nvenc",  # Use NVENC encoder with CUDA
                    "-preset",
                    "fast",  # Use faster preset to reduce memory usage
                    "-rc",
                    "constqp",  # Use constant quantization parameter mode
                    "-qp",
                    "23",  # Quality level (lower is better but uses more memory)
                    "-c:a",
                    "aac",  # Use AAC for audio
                    output_path,
                ]

                print(
                    f"Running FFmpeg command with optimized settings: {' '.join(cmd)}"
                )
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError as e:
                print(f"Optimized CUDA encoding failed: {str(e)}")

                # Try with simpler CUDA acceleration settings
                try:
                    # Simple NVENC encoding with fewer options
                    cmd = [
                        "ffmpeg",
                        "-y",  # Overwrite output file if it exists
                        "-ss",
                        str(start_time),  # FAST SEEK: Place -ss before -i
                        "-i",
                        input_path,
                        "-t",
                        str(duration),
                        "-c:v",
                        "h264_nvenc",  # Use NVENC encoder with CUDA
                        "-c:a",
                        "aac",  # Use AAC for audio
                        output_path,
                    ]

                    print(
                        f"Running FFmpeg command with simple NVENC encoding: {' '.join(cmd)}"
                    )
                    subprocess.run(cmd, check=True)
                    return
                except subprocess.CalledProcessError as e:
                    print(f"Simple NVENC encoding failed: {str(e)}")

                    # Try with full hardware acceleration pipeline as last CUDA attempt
                    try:
                        cmd = [
                            "ffmpeg",
                            "-y",  # Overwrite output file if it exists
                            "-ss",
                            str(start_time),  # FAST SEEK: Place -ss before -i
                            *hw_accel_cmd,  # Place hardware acceleration before input
                            "-i",
                            input_path,
                            "-t",
                            str(duration),
                            "-c:v",
                            "h264_nvenc",  # Use NVENC encoder with CUDA
                            "-c:a",
                            "aac",  # Use AAC for audio
                            output_path,
                        ]

                        print(
                            f"Running FFmpeg command with full CUDA pipeline: {' '.join(cmd)}"
                        )
                        subprocess.run(cmd, check=True)
                        return
                    except subprocess.CalledProcessError as e:
                        print(f"Full CUDA pipeline failed: {str(e)}")

        # If we reach here, either no hardware acceleration is available or it failed
        # Fall back to software encoding with OPTIMIZED seeking
        print("Using software encoding for extraction with fast seeking")
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-ss",
            str(start_time),  # CRITICAL: -ss BEFORE -i for fast seeking
            "-accurate_seek",  # Ensure accurate keyframe seeking
            "-i",
            input_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",  # Software encoding
            "-preset",
            "ultrafast",  # Use ultrafast preset for 2-3x faster encoding
            "-crf",
            "23",  # Constant quality
            "-c:a",
            "aac",
            "-b:a",
            "128k",  # Audio bitrate
            output_path,
        ]

        print(f"Running OPTIMIZED FFmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except Exception as e:
        print(f"Error extracting clip: {str(e)}")
        import traceback

        print(traceback.format_exc())

        # Last resort fallback - try with absolute minimum settings
        try:
            print("Attempting last resort extraction with minimal settings")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),  # FAST SEEK: Place -ss before -i
                "-i",
                input_path,
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",  # Fastest preset
                "-c:a",
                "aac",
                output_path,
            ]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as fallback_error:
            print(f"All extraction attempts failed: {str(fallback_error)}")
        raise


def get_hardware_acceleration_cmd():
    """Get FFmpeg hardware acceleration command based on available hardware.

    Returns:
        List of FFmpeg command arguments for hardware acceleration
    """
    try:
        # Check for NVIDIA GPU
        try:
            nvidia_smi_output = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if nvidia_smi_output.returncode == 0:
                print("[INFO] NVIDIA GPU detected, using NVENC hardware acceleration")
                # For NVIDIA GPUs, use these hardware acceleration parameters
                # First check if h264_nvenc encoder is available
                encoders_output = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if "h264_nvenc" in encoders_output.stdout:
                    print("[INFO] NVENC encoder is available")
                    return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                else:
                    print(
                        "[WARNING] NVENC encoder not available despite NVIDIA GPU being detected"
                    )
        except (subprocess.SubprocessError, TimeoutError) as e:
            print(f"[INFO] Error checking for NVIDIA GPU: {str(e)}")

        # Check for AMD GPU (Windows)
        if os.name == "nt":
            try:
                amd_check = subprocess.run(
                    ["dxdiag", "/t"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                if "AMD" in amd_check.stdout:
                    print("[INFO] AMD GPU detected, using AMF hardware acceleration")
                    # Check if AMD encoder is available
                    encoders_output = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-encoders"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if "h264_amf" in encoders_output.stdout:
                        return ["-hwaccel", "amf"]
                    else:
                        print(
                            "[WARNING] AMF encoder not available despite AMD GPU being detected"
                        )
            except (subprocess.SubprocessError, TimeoutError):
                pass

        # Check for Intel QuickSync
        try:
            # Different ways to check for Intel CPU/GPU
            if os.name == "nt":  # Windows
                cpu_info = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                is_intel = "Intel" in cpu_info.stdout
            else:  # Linux
                try:
                    cpu_info = subprocess.run(
                        ["lscpu"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5,
                    )
                    is_intel = "Intel" in cpu_info.stdout
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Try another method if lscpu is not available
                    with open("/proc/cpuinfo", "r") as f:
                        is_intel = "Intel" in f.read()

            if is_intel:
                print(
                    "[INFO] Intel CPU detected, trying QuickSync hardware acceleration"
                )
                # Check if QSV encoder is available
                encoders_output = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if "h264_qsv" in encoders_output.stdout:
                    return ["-hwaccel", "qsv"]
                else:
                    print(
                        "[WARNING] QSV encoder not available despite Intel CPU being detected"
                    )
        except Exception as e:
            print(f"[INFO] Error checking for Intel CPU: {str(e)}")

        print(
            "[INFO] No compatible GPU detected for hardware acceleration, using software encoding"
        )
        return []
    except Exception as e:
        print(
            f"[WARNING] Error detecting hardware acceleration: {str(e)}. Using software encoding."
        )
        return []


def convert_to_vertical(input_path: str, output_path: str) -> None:
    """Convert video to vertical (9:16) format using a simple center crop or letterbox.

    This simplified version does not use face-tracking. It calculates a center crop
    (or pads if needed) and performs a single ffmpeg pass to produce the output.
    """
    try:
        # Get video properties
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Target portrait dimensions (1080x1920)
        target_width = 1080
        target_height = 1920

        # Compute crop width to preserve full height and achieve 9:16 (width = height * 9/16)
        crop_width = int(height * 9 / 16)
        if crop_width % 2 != 0:
            crop_width -= 1

        # Bound crop width to video width
        if crop_width > width:
            # Not enough width: we'll letterbox/pad instead of crop
            vf = f"scale=iw*min({target_width}/iw\,{target_height}/ih):ih*min({target_width}/iw\,{target_height}/ih),pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
        else:
            # Center crop horizontally then scale/pad to target
            x_offset = max(0, (width - crop_width) // 2)
            # Ensure even offsets
            if x_offset % 2 != 0:
                x_offset -= 1
            vf = f"crop={crop_width}:{height}:{x_offset}:0,scale=iw*min({target_width}/iw\,{target_height}/ih):ih*min({target_width}/iw\,{target_height}/ih),pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"

        # Run a single ffmpeg pass to perform crop/scale/pad
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            output_path,
        ]

        print(f"Running vertical conversion (center crop/letterbox): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except Exception as e:
        print(f"Error in convert_to_vertical: {str(e)}")
        import traceback

        print(traceback.format_exc())
        raise


def add_captions_and_game(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    """Add captions and game overlay to the video.

    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        start_time: Start time of the clip in seconds
        end_time: End time of the clip in seconds
    """
    try:
        # Get video dimensions
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create temporary files for overlays
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as caption_file:
            caption_path = caption_file.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as game_file:
            game_path = game_file.name

        # Create caption and game overlay images
        create_caption_image(caption_path, "Sample Caption Text", width, height)
        create_game_overlay(game_path, width, height)

        # Get hardware acceleration command
        hw_accel_cmd = get_hardware_acceleration_cmd()

        # Calculate overlay positions
        # Pre-calculate exact pixel positions instead of using expressions
        caption_y_position = height - 200  # 200 pixels from bottom for caption
        game_y_position = height - 100  # 100 pixels from bottom for game overlay

        try:
            # Try with hardware acceleration if available
            if hw_accel_cmd and "cuda" in " ".join(hw_accel_cmd):
                # For CUDA, use hardware decoding but CPU processing and hardware encoding
                # This avoids format compatibility issues with overlays
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hwaccel",
                    "cuda",  # Use CUDA for decoding only
                    "-i",
                    input_path,
                    "-i",
                    caption_path,
                    "-i",
                    game_path,
                    "-filter_complex",
                    f"[0:v][1:v]overlay=0:{caption_y_position}[v1];[v1][2:v]overlay=0:{game_y_position}[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "0:a",
                    "-c:v",
                    "h264_nvenc",  # Use NVENC encoder
                    "-preset",
                    "p1",
                    "-c:a",
                    "copy",
                    output_path,
                ]

                print(f"Running overlay command with CUDA encoder: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                # Without hardware acceleration, use software encoding
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-i",
                    caption_path,
                    "-i",
                    game_path,
                    "-filter_complex",
                    f"[0:v][1:v]overlay=0:{caption_y_position}[v1];[v1][2:v]overlay=0:{game_y_position}[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "0:a",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "copy",
                    output_path,
                ]

                print(f"Running overlay command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error with overlay processing: {e}")
            print("Trying fallback overlay method")

            # Fallback to a simpler overlay method with software encoding
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-i",
                caption_path,
                "-filter_complex",
                f"[0:v][1:v]overlay=0:{caption_y_position}[v]",
                "-map",
                "[v]",
                "-map",
                "0:a",
                "-c:v",
                "libx264",
                "-c:a",
                "copy",
                output_path,
            ]

            print(f"Running simplified overlay command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # Clean up temporary files
        os.unlink(caption_path)
        os.unlink(game_path)

    except Exception as e:
        print(f"Error adding captions and game overlay: {str(e)}")
        import traceback

        print(traceback.format_exc())
        raise


def create_caption_image(output_path: str, text: str, width: int, height: int) -> None:
    """Create a caption overlay image.

    Args:
        output_path: Path to save the output image
        text: Caption text
        width: Width of the image
        height: Height of the image
    """
    # Create a transparent image
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Add a semi-transparent background
    draw.rectangle([(0, 0), (width, height)], fill=(0, 0, 0, 128))

    # Add text
    # In a real application, you would use a proper font and size
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    # Center the text
    # Using textbbox instead of deprecated textsize
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    position = ((width - text_width) // 2, (height - text_height) // 2)

    # Draw the text
    draw.text(position, text, fill=(255, 255, 255, 255), font=font)

    # Save the image
    image.save(output_path)


def create_game_overlay(output_path: str, width: int, height: int) -> None:
    """Create a game overlay image.

    Args:
        output_path: Path to save the output image
        width: Width of the image
        height: Height of the image
    """
    # Create a transparent image
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Add a semi-transparent background
    draw.rectangle([(0, 0), (width, height)], fill=(0, 0, 0, 128))

    # Add game elements (simplified)
    # In a real application, you would add actual game elements
    # Center the title text properly
    title_text = "Tap the emoji game!"
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        title_font = ImageFont.load_default()

    # Get text dimensions using textbbox
    left, top, right, bottom = draw.textbbox((0, 0), title_text, font=title_font)
    text_width = right - left
    # Center horizontally
    x_position = (width - text_width) // 2
    draw.text(
        (x_position, height // 2),
        title_text,
        fill=(255, 255, 255, 255),
        font=title_font,
    )

    # Add some emoji as game elements
    emojis = ["😀", "😂", "🎉", "🔥", "👍"]
    try:
        emoji_font = ImageFont.truetype("arial.ttf", 48)
    except IOError:
        emoji_font = ImageFont.load_default()

    for i, emoji in enumerate(emojis):
        x = width // (len(emojis) + 1) * (i + 1)
        y = height // 2 + 50
        draw.text((x, y), emoji, fill=(255, 255, 255, 255), font=emoji_font)

    # Save the image
    image.save(output_path)


async def generate_thumbnail(
    video_path: str, output_path: str, timestamp: float = None
) -> str:
    """
    Generate a thumbnail image from a video at a specific timestamp, formatted as 1080x1920 portrait with black bars (letterboxing) on top and bottom, preserving the full width of the original frame.
    """
    import os
    import subprocess

    # Get video duration using ffprobe to ensure timestamp is valid
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )

    try:
        duration = float(result.stdout.strip())
        print(f"Video duration: {duration} seconds")
        if timestamp is None:
            timestamp = duration / 3  # Use 1/3 of the video duration
        if timestamp >= duration:
            print(f"Timestamp {timestamp} is beyond video duration {duration}, adjusting...")
            timestamp = max(0, min(duration - 0.5, duration / 2))
            print(f"Adjusted timestamp to {timestamp}")
    except (ValueError, TypeError) as e:
        print(f"Error parsing duration: {str(e)}, using default timestamp of 0")
        timestamp = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    abs_output_path = os.path.abspath(output_path)

    # Target portrait resolution
    target_width = 1080
    target_height = 1920
    filter_str = (
        f"scale=iw*min({target_width}/iw\,{target_height}/ih):ih*min({target_width}/iw\,{target_height}/ih),"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
    )

    # FFmpeg command to extract and format the thumbnail
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", filter_str,
        "-q:v", "2",
        abs_output_path,
    ]
    print(f"Generating portrait thumbnail with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    if os.path.exists(abs_output_path) and os.path.getsize(abs_output_path) > 0:
        print(f"Thumbnail generated successfully: {abs_output_path}")
        return abs_output_path
    else:
        raise RuntimeError(f"Failed to generate thumbnail at {abs_output_path}")


