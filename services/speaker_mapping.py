"""
Optimized Speaker Mapping Strategy

Instead of running TalkNet on every frame (CPU-intensive), we:
1. Run TalkNet ONLY on first 5-10 seconds to identify speaker positions
2. Map speaker labels (A, B, C) from transcription to X coordinates
3. Use transcription timestamps to control cropping dynamically
4. If new speaker appears, do quick face detection to find position

This reduces computation by 90%+ while maintaining accuracy.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


async def map_speakers_from_sample(
    video_path: str,
    transcript: Dict[str, Any],
    sample_duration: float = 10.0
) -> Dict[str, int]:
    """
    Analyze first N seconds of video to map speaker labels to X positions.
    Uses YOLO8 for face detection (GPU-accelerated) + TalkNet for speaker identification.
    
    Args:
        video_path: Path to video file
        transcript: Full transcript with speaker labels
        sample_duration: How many seconds to analyze (default 10s)
    
    Returns:
        Dictionary mapping speaker labels to X coordinates
        Example: {"A": 400, "B": 1200, "C": 800}
    """
    from services.talknet_asd import detect_active_speaker_simple, TALKNET_AVAILABLE
    from services.yolo_face_detection import detect_faces, get_device
    
    logger.info(f"🎯 Mapping speakers using first {sample_duration}s of video...")
    logger.info(f"   Using YOLO8 face detection on {get_device().upper()}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract speaker utterances from transcript in sample window
    utterances = transcript.get('utterances', [])
    sample_utterances = [
        u for u in utterances 
        if u.get('start', 0) / 1000 < sample_duration  # AssemblyAI uses milliseconds
    ]
    
    if not sample_utterances:
        logger.warning("⚠️ No utterances found in sample window, using full video detection")
        cap.release()
        return await _fallback_visual_mapping(video_path, transcript)
    
    logger.info(f"   Found {len(sample_utterances)} utterances in sample window")
    
    # STEP 1: Run YOLO8 face detection on sample frames
    logger.info(f"   📷 Detecting faces in sample window using YOLO8...")
    person_detections = []
    
    frame_idx = 0
    max_sample_frames = int(sample_duration * fps)
    
    # Collect frames for batch processing (better GPU utilization)
    frames_batch = []
    frame_indices = []
    
    while cap.isOpened() and frame_idx < max_sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every 5th frame for efficiency
        if frame_idx % 5 == 0:
            frames_batch.append(frame)
            frame_indices.append(frame_idx)
            
            # Process in batches of 8 frames for optimal GPU usage
            if len(frames_batch) == 8:
                from services.yolo_face_detection import detect_faces_batch_yolo
                batch_detections = detect_faces_batch_yolo(frames_batch)
                
                for batch_idx, frame_detections in enumerate(batch_detections):
                    current_frame_idx = frame_indices[batch_idx]
                    for detection in frame_detections:
                        person_detections.append({
                            'frame': current_frame_idx,
                            'x': detection['center_x'],
                            'y': detection['center_y'],
                            'width': detection['face_width'],
                            'height': detection['face_height'],
                            'time': current_frame_idx / fps,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence']
                        })
                
                frames_batch = []
                frame_indices = []
        
        frame_idx += 1
    
    # Process remaining frames
    if frames_batch:
        from services.yolo_face_detection import detect_faces_batch_yolo
        batch_detections = detect_faces_batch_yolo(frames_batch)
        
        for batch_idx, frame_detections in enumerate(batch_detections):
            current_frame_idx = frame_indices[batch_idx]
            for detection in frame_detections:
                person_detections.append({
                    'frame': current_frame_idx,
                    'x': detection['center_x'],
                    'y': detection['center_y'],
                    'width': detection['face_width'],
                    'height': detection['face_height'],
                    'time': current_frame_idx / fps,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                })
    
    cap.release()
    
    logger.info(f"   ✅ Detected {len(person_detections)} person instances")
    
    # If no faces detected in first 10s, try a longer sample or middle of video
    if len(person_detections) == 0 and sample_duration < 30:
        logger.warning(f"   ⚠️ No faces in first {sample_duration}s, trying first 30 seconds...")
        # Retry with 30 seconds
        return await map_speakers_from_sample(video_path, transcript, sample_duration=30.0)
    
    # STEP 2: Run TalkNet on sample window ONLY
    if TALKNET_AVAILABLE and person_detections:
        logger.info(f"   🎙️ Running TalkNet ASD on sample window (this is the ONLY time we run it!)...")
        
        # Create temporary video file for sample
        import tempfile
        temp_sample_path = tempfile.mktemp(suffix='.mp4')
        
        # Extract sample clip
        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-t', str(sample_duration),
            '-c', 'copy',
            temp_sample_path,
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True)
        
        # Run TalkNet on sample
        speaker_scores = await detect_active_speaker_simple(
            temp_sample_path,
            person_detections,
            fps
        )
        
        # Clean up temp file
        import os
        if os.path.exists(temp_sample_path):
            os.remove(temp_sample_path)
        
        # Add TalkNet scores to detections
        scores_added = 0
        for detection in person_detections:
            frame_num = detection['frame']
            if frame_num in speaker_scores:
                # Find closest X position in this frame's scores
                frame_x_scores = speaker_scores[frame_num]
                if frame_x_scores:
                    closest_x = min(frame_x_scores.keys(), key=lambda x: abs(x - detection['x']))
                    detection['talknet_score'] = frame_x_scores[closest_x]
                    scores_added += 1
        
        logger.info(f"   ✅ TalkNet analysis complete on sample")
        logger.info(f"   📊 Added TalkNet scores to {scores_added}/{len(person_detections)} detections")
        
        # Debug: Show score distribution
        all_scores = [d.get('talknet_score') for d in person_detections if 'talknet_score' in d]
        if all_scores:
            logger.info(f"   📊 Score range: min={min(all_scores):.3f}, max={max(all_scores):.3f}, mean={np.mean(all_scores):.3f}")
    
    # STEP 3: Map speaker labels to X positions
    logger.info(f"   🗺️ Mapping speaker labels to positions...")
    
    # Debug: Check utterance format
    if sample_utterances:
        logger.info(f"   🔍 Sample utterance keys: {list(sample_utterances[0].keys())}")
        logger.info(f"   🔍 First utterance: {sample_utterances[0]}")
    
    speaker_positions = {}  # label -> list of X positions
    
    for utterance in sample_utterances:
        speaker = utterance.get('speaker')
        if not speaker:
            continue
        
        # Get start/end time (already in seconds from AssemblyAI)
        start_time = utterance.get('start', 0)
        end_time = utterance.get('end', 0)
        
        # Find detections during this utterance
        utterance_detections = [
            d for d in person_detections
            if start_time <= d['time'] <= end_time
        ]
        
        if not utterance_detections:
            continue
        
        # Find the person who was speaking (highest TalkNet score or most activity)
        if TALKNET_AVAILABLE:
            # Find detections with TalkNet scores
            scored_detections = [
                d for d in utterance_detections
                if 'talknet_score' in d
            ]
            if scored_detections:
                # Use the detection with highest score (can be negative, that's OK)
                best_detection = max(scored_detections, key=lambda d: d.get('talknet_score', float('-inf')))
            else:
                # No TalkNet scores available, use center-most person
                best_detection = utterance_detections[len(utterance_detections) // 2]
        else:
            # Without TalkNet, use the most consistently detected person
            x_positions = [d['x'] for d in utterance_detections]
            median_x = int(np.median(x_positions))
            best_detection = min(utterance_detections, key=lambda d: abs(d['x'] - median_x))
        
        # Add to speaker positions
        if speaker not in speaker_positions:
            speaker_positions[speaker] = []
        speaker_positions[speaker].append(best_detection['x'])
    
    # Average positions for each speaker
    speaker_map = {}
    for speaker, positions in speaker_positions.items():
        speaker_map[speaker] = int(np.median(positions))
        logger.info(f"      {speaker}: X = {speaker_map[speaker]}")
    
    logger.info(f"   ✅ Mapped {len(speaker_map)} speakers")
    
    return speaker_map


async def _fallback_visual_mapping(
    video_path: str,
    transcript: Dict[str, Any]
) -> Dict[str, int]:
    """
    Fallback: If no utterances in sample window, detect all unique speakers visually using YOLO8.
    Returns mapping of detected positions to generic labels.
    """
    import cv2
    from services.yolo_face_detection import detect_faces
    
    logger.info("   Using fallback visual mapping with YOLO8...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # Detect all faces in first 100 frames
    x_positions = []
    
    for frame_idx in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 10 == 0:
            detections = detect_faces(frame)
            
            for detection in detections:
                x_positions.append(detection['center_x'])
    
    cap.release()
    
    # Cluster X positions to find unique speakers
    if x_positions:
        sorted_x = sorted(set(x_positions))
        # Group positions within 200px as same person
        speakers = []
        current_group = [sorted_x[0]]
        
        for x in sorted_x[1:]:
            if x - current_group[-1] < 200:
                current_group.append(x)
            else:
                speakers.append(int(np.mean(current_group)))
                current_group = [x]
        
        if current_group:
            speakers.append(int(np.mean(current_group)))
        
        # Map to labels A, B, C, etc.
        speaker_map = {
            chr(65 + i): x
            for i, x in enumerate(sorted(speakers))
        }
        
        logger.info(f"   Found {len(speaker_map)} unique speaker positions")
        return speaker_map
    
    return {}


def generate_crop_positions_from_transcript(
    transcript: Dict[str, Any],
    speaker_map: Dict[str, int],
    start_time: float,
    end_time: float,
    fps: float,
    input_w: int,
    crop_w: int
) -> List[Tuple[int, int]]:
    """
    Generate crop positions based on transcript timing and speaker map.
    
    This is MUCH faster than frame-by-frame analysis!
    
    Args:
        transcript: Full transcript with utterances
        speaker_map: Mapping of speaker labels to X coordinates
        start_time: Clip start time
        end_time: Clip end time
        fps: Video frame rate
        input_w: Input video width
        crop_w: Crop width
    
    Returns:
        List of (frame_number, center_x) tuples
    """
    logger.info(f"   📊 Generating crop positions from transcript (no heavy computation!)...")
    
    utterances = transcript.get('utterances', [])
    
    # Filter utterances to clip time range (times are already in seconds from AssemblyAI)
    clip_utterances = [
        u for u in utterances
        if u.get('start', 0) < end_time and u.get('end', 0) > start_time
    ]
    
    if not clip_utterances:
        logger.warning("   No utterances in clip range, using center crop")
        return [(int((end_time - start_time) * fps), input_w // 2)]
    
    crop_positions = []
    current_x = None
    
    # For each frame in the clip
    total_frames = int((end_time - start_time) * fps)
    
    for frame_num in range(total_frames):
        frame_time = start_time + (frame_num / fps)
        
        # Find which speaker is talking at this time
        active_speaker = None
        for utterance in clip_utterances:
            utt_start = utterance.get('start', 0)  # Already in seconds from AssemblyAI
            utt_end = utterance.get('end', 0)      # Already in seconds from AssemblyAI
            
            if utt_start <= frame_time <= utt_end:
                active_speaker = utterance.get('speaker')
                break
        
        if active_speaker and active_speaker in speaker_map:
            target_x = speaker_map[active_speaker]
        elif active_speaker:
            # Unknown speaker - log warning and use center (will trigger fallback in parent)
            if frame_num == 0:  # Only log once per clip
                logger.warning(f"   🚨 UNMAPPED SPEAKER DETECTED: '{active_speaker}' at {frame_time:.2f}s")
                logger.warning(f"      📋 Currently mapped speakers: {list(speaker_map.keys())}")
                logger.warning(f"      🔄 Triggering speaker discovery mode for this clip")
            target_x = None  # Signal that we need fallback
            break  # Exit early - this clip needs full analysis
        elif current_x is not None:
            target_x = current_x  # Stay with last position
        else:
            target_x = input_w // 2  # Default to center
        
        # Clamp to valid range
        half_crop = crop_w // 2
        target_x = max(half_crop, min(input_w - half_crop, target_x))
        
        current_x = target_x
        crop_positions.append((frame_num, target_x))
    
    # Check if we exited early due to unmapped speaker
    if crop_positions and crop_positions[-1][1] is None:
        logger.warning(f"   ⚠️ Clip contains unmapped speakers - returning empty to trigger fallback")
        return []  # Empty list signals fallback needed
    
    logger.info(f"   ✅ Generated {len(crop_positions)} crop positions from transcript")
    
    return crop_positions
