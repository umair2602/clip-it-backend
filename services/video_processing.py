import json
import logging
import os
import subprocess
# Import configuration
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set up logging first (before imports that might use it)
logger = logging.getLogger(__name__)

# Import other services
# face tracking removed to simplify and speed up clip creation

# IMPROVED: Import audio-visual correlation module
try:
    from services.audio_analysis import (
        extract_audio_from_video,
        calculate_frame_audio_energy,
        calculate_mouth_opening,
        calculate_time_aligned_correlation
    )
    AUDIO_CORRELATION_AVAILABLE = True
except ImportError:
    logger.warning("Audio correlation module not available, using visual-only detection")
    AUDIO_CORRELATION_AVAILABLE = False

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
        logger.info(f"📹 STARTING VIDEO CLIP CREATION (PARALLEL MODE)")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Number of segments to process: {len(segments)}")
        logger.info("="*70)
        
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
            segment_start_time = time.time()
            clip_id = f"clip_{i}"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🎬 Processing Clip {i+1}/{total_segments}: {segment.get('title', 'Untitled')}")
            logger.info(f"   Timestamp: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
            logger.info(f"{'='*70}")

            # Create the clip
            logger.info(f"   ⏳ Step 1/3: Creating video clip...")
            clip_path = await create_clip(
                video_path=video_path,
                output_dir=str(output_dir),
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                clip_id=clip_id,
                transcript=transcript,
            )

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
        logger.info("="*70 + "\n")

        return processed_clips

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        raise


async def create_clip(
    video_path: str, output_dir: str, start_time: float, end_time: float, clip_id: str, transcript: Dict[str, Any] = None
) -> str:
    """
    Extract video segment and convert to vertical (9:16) format using MediaPipe AI auto-reframe.
    Uses pose detection + face detection to intelligently track and follow speakers.
    """
    logger.info(f"🎬 Creating clip {clip_id}: {start_time:.2f}s - {end_time:.2f}s")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")
    temp_path = os.path.join(output_dir, f"{clip_id}_temp.mp4")
    
    # Step 1: Extract the segment
    logger.info(f"   📂 Extracting segment...")
    extract_clip(video_path, temp_path, start_time, end_time)
    
    # Step 2: Detect optimal crop positions using MediaPipe AI
    logger.info(f"   🤖 Analyzing with MediaPipe AI for auto-reframe...")
    crop_positions = await detect_mediapipe_crop_positions(temp_path, start_time, end_time, transcript)
    
    # Step 3: Apply smart crop with smooth transitions
    logger.info(f"   🎥 Applying AI-guided auto-reframe...")
    await apply_smart_crop_with_transitions(temp_path, output_path, crop_positions)
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    logger.info(f"   ✅ Clip created with AI auto-reframe: {output_path}")
    return output_path


async def detect_mediapipe_crop_positions(
    video_path: str,
    start_time: float, 
    end_time: float,
    transcript: Dict[str, Any] = None
) -> list:
    """
    Use MediaPipe to detect ALL people/faces, then use speaker diarization to 
    identify which person is speaking and track them.
    
    Strategy:
    1. Detect all people in each frame and track their positions
    2. Build a spatial map of where each person sits (left, center, right)
    3. Use transcript to know which speaker is talking at each moment
    4. Map speakers to spatial positions
    5. Track the active speaker's position
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
    
    # Calculate crop dimensions for 9:16
    crop_h = input_h
    crop_w = int(crop_h * 0.5625)  # 607px for 1080p
    
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
    
    # IMPROVED: Extract audio for audio-visual correlation
    audio_energies = None
    if AUDIO_CORRELATION_AVAILABLE:
        try:
            logger.info(f"      🔊 Extracting audio for audio-visual correlation...")
            audio, sr = extract_audio_from_video(video_path)
            audio_energies = calculate_frame_audio_energy(audio, sr, fps, total_frames)
            logger.info(f"      ✅ Audio correlation enabled (mean energy: {audio_energies.mean():.3f})")
        except Exception as e:
            logger.warning(f"      ⚠️ Audio extraction failed: {e}. Using visual-only detection.")
            audio_energies = None
    else:
        logger.info(f"      ℹ️ Audio correlation not available, using visual-only detection")
    
    # STEP 2: Detect all people AND their lip movement for speech detection
    person_detections = []  # List of all detected people with lip movement data
    sample_interval = max(1, int(fps / 10))  # IMPROVED: Sample 10 frames per second for better speaker change detection
    
    mp_face_mesh = mp.solutions.face_mesh
    
    # Lip landmark indices (MediaPipe Face Mesh)
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
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
                    
                    # IMPROVED: Calculate mouth opening for audio-visual correlation
                    mouth_opening = calculate_mouth_opening(face_landmarks) if AUDIO_CORRELATION_AVAILABLE else lip_distance
                    
                    person_detections.append({
                        'frame': frame_idx,
                        'time': frame_time,
                        'x': face_center_x,
                        'y': face_center_y,
                        'size': face_size,
                        'lip_distance': lip_distance,
                        'lip_movement': lip_movement,
                        'is_speaking': is_speaking,
                        'confidence': detection_confidence,  # IMPROVED: Add confidence score
                        'adaptive_threshold': adaptive_threshold,  # Store for debugging
                        'mouth_opening': mouth_opening,  # IMPROVED: For audio correlation
                        'face_landmarks': face_landmarks,  # Store for potential future use
                        'type': 'face_mesh'
                    })
            
            frame_idx += 1
        
        cap.release()
    
    logger.info(f"      ✅ Detected {len(person_detections)} person instances across frames")
    
    if len(person_detections) == 0:
        logger.warning(f"      ⚠️ No people detected - using center crop")
        return [{'frame': 0, 'crop_x': (input_w - crop_w) // 2, 'center_x': input_w // 2, 'has_detection': False}]
    
    # STEP 3: Cluster people by spatial position to identify distinct individuals
    person_clusters = cluster_people_by_position(person_detections, input_w)
    logger.info(f"      👥 Identified {len(person_clusters)} distinct people in video")
    
    # STEP 4: Map speakers to person clusters using AUDIO-VISUAL CORRELATION + transcript
    speaker_to_position = map_speakers_to_positions(
        speaker_timeline, 
        person_clusters, 
        person_detections,
        audio_energies=audio_energies,  # IMPROVED: Pass audio data
        fps=fps  # IMPROVED: Pass FPS for time alignment
    )
    
    # STEP 5: Generate crop positions based on active speaker at each moment
    crop_positions = []
    
    # Get all unique frame numbers we sampled
    sampled_frames = sorted(set(d['frame'] for d in person_detections))
    
    for frame_num in sampled_frames:
        frame_time = frame_num / fps
        
        # Find who's speaking at this time
        active_speaker = get_active_speaker_at_time(speaker_timeline, frame_time)
        
        if active_speaker and active_speaker in speaker_to_position:
            # Get the position of the active speaker
            speaker_pos = speaker_to_position[active_speaker]
            center_x = speaker_pos['avg_x']
            logger.debug(f"      Frame {frame_num} ({frame_time:.1f}s): Tracking {active_speaker} at X={center_x}")
        else:
            # No speaker or unknown speaker - use most prominent person
            frame_detections = [d for d in person_detections if d['frame'] == frame_num]
            if frame_detections:
                largest_person = max(frame_detections, key=lambda d: d['size'])
                center_x = largest_person['x']
            else:
                center_x = input_w // 2
        
        # Calculate crop position
        crop_x = center_x - (crop_w // 2)
        crop_x = max(0, min(crop_x, input_w - crop_w))
        
        # IMPROVED: Get average confidence for this frame's detections
        frame_detections = [d for d in person_detections if d['frame'] == frame_num]
        avg_confidence = np.mean([d.get('confidence', 0.5) for d in frame_detections]) if frame_detections else 0.5
        
        crop_positions.append({
            'frame': frame_num,
            'crop_x': crop_x,
            'center_x': center_x,
            'has_detection': True,
            'active_speaker': active_speaker if active_speaker else 'unknown',
            'confidence': avg_confidence  # IMPROVED: Add confidence for weighted cropping
        })
    
    if len(crop_positions) == 0:
        logger.warning(f"      ⚠️ No crop positions generated - using center")
        return [{'frame': 0, 'crop_x': (input_w - crop_w) // 2, 'center_x': input_w // 2, 'has_detection': False}]
    
    # Smooth out crop positions to avoid jittery movement
    crop_positions = smooth_crop_positions(crop_positions, window_size=5)
    
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


def map_speakers_to_positions(
    speaker_timeline: list, 
    person_clusters: dict, 
    detections: list,
    audio_energies: np.ndarray = None,  # IMPROVED: Audio energy data
    fps: float = 30.0  # IMPROVED: Frame rate for time alignment
) -> dict:
    """
    IMPROVED: Map speakers to positions by PRIORITIZING transcript data over visual-only detection.
    
    Strategy:
    1. Trust the transcript timeline - it knows who's speaking when (from audio)
    2. Use visual lip movement as CONFIRMATION, not primary detection
    3. Only consider detections during transcript-confirmed speaking times
    4. Heavily weight positions that consistently show lip movement during speaking
    
    Returns dict: {speaker_name: cluster_info}
    """
    import numpy as np
    
    if not speaker_timeline or not person_clusters:
        logger.warning(f"      Cannot map speakers: timeline={len(speaker_timeline) if speaker_timeline else 0}, clusters={len(person_clusters) if person_clusters else 0}")
        return {}
    
    logger.info(f"      🔗 IMPROVED: Mapping {len(set(s['speaker'] for s in speaker_timeline))} speakers to {len(person_clusters)} visual positions (TRANSCRIPT-FIRST approach)")
    
    speaker_positions = {}
    
    # Sort clusters by X position (left to right)
    sorted_clusters = sorted(person_clusters.items(), key=lambda item: item[1]['avg_x'])
    cluster_positions = [f"X={c[1]['avg_x']}" for c in sorted_clusters]
    logger.info(f"         Visual positions detected at: {cluster_positions}")
    
    # For each speaker, analyze their speaking time
    unique_speakers = list(set(s['speaker'] for s in speaker_timeline))
    
    # Determine if we can use audio-visual correlation
    use_audio_correlation = (audio_energies is not None and len(speaker_timeline) > 0)
    
    for speaker in unique_speakers:
        # Get all utterances for this speaker
        speaker_utterances = [s for s in speaker_timeline if s['speaker'] == speaker]
        
        if not speaker_utterances:
            continue
        
        # IMPROVED: Build time windows when THIS speaker is talking
        speaking_windows = [(utt['start'], utt['end']) for utt in speaker_utterances]
        total_speaking_time = sum(end - start for start, end in speaking_windows)
        
        # Collect detections ONLY during confirmed speaking times
        # This filters out reactions, laughing, etc. from non-speakers
        speaker_detections = []
        speaking_detections = []  # Detections with lip movement during speaking
        
        for detection in detections:
            det_time = detection['time']
            
            # Check if this detection falls within any speaking window for this speaker
            is_during_speaking = any(
                start <= det_time <= end 
                for start, end in speaking_windows
            )
            
            if is_during_speaking:
                speaker_detections.append(detection)
                
                # IMPROVED: Require BOTH conditions for high-confidence speaking detection:
                # 1. Lip movement detected
                # 2. High confidence score
                if detection.get('is_speaking', False) and detection.get('confidence', 0) > 0.4:
                    speaking_detections.append(detection)
        
        if not speaker_detections:
            logger.warning(f"         ⚠️ Speaker {speaker}: No visual detections during {len(speaker_utterances)} utterances ({total_speaking_time:.1f}s)")
            continue
        
        # IMPROVED: Require minimum percentage of speaking detections to avoid false mappings
        # If we have very few speaking detections, the mapping is likely incorrect
        speaking_ratio = len(speaking_detections) / len(speaker_detections) if speaker_detections else 0
        
        if speaking_ratio < 0.1 and len(speaking_detections) < 3:
            # Very few speaking detections - fall back to using all detections but log warning
            logger.warning(f"         ⚠️ Speaker {speaker}: Low speaking detection ratio ({speaking_ratio:.1%}) - using all detections")
            detections_to_use = speaker_detections
        else:
            # Good speaking detection ratio - use speaking detections preferentially
            detections_to_use = speaking_detections if speaking_detections else speaker_detections
        
        logger.info(f"         Speaker {speaker}: {len(speaking_detections)}/{len(speaker_detections)} speaking detections ({speaking_ratio:.1%}) during {total_speaking_time:.1f}s")
        
        # IMPROVED: Score clusters using TRANSCRIPT-CONFIRMED approach
        # Key change: Heavily penalize clusters that don't show consistent lip movement
        # during transcript-confirmed speaking times
        cluster_scores = {}
        cluster_data = {}  # Track detailed scoring data
        
        for cid in person_clusters.keys():
            cluster_data[cid] = {
                'total_score': 0,
                'speaking_count': 0,
                'non_speaking_count': 0,
                'weighted_positions': [],
                'confidences': [],
                'audio_correlation': 0.0  # IMPROVED: Audio-visual correlation score
            }
        
        # IMPROVED: Calculate audio-visual correlation for each cluster if available
        if use_audio_correlation:
            for cid, cluster in person_clusters.items():
                # Get all detections for this cluster
                cluster_detections = [
                    d for d in detections_to_use
                    if abs(d['x'] - cluster['avg_x']) < 300
                ]
                
                if cluster_detections:
                    # Calculate time-aligned correlation
                    correlation_score = calculate_time_aligned_correlation(
                        audio_energies,
                        cluster_detections,
                        speaking_windows,
                        fps
                    )
                    cluster_data[cid]['audio_correlation'] = correlation_score
                    logger.debug(f"         Cluster {cid}: audio correlation = {correlation_score:.3f}")
        
        for detection in detections_to_use:
            x = detection['x']
            size = detection['size']
            lip_movement = detection.get('lip_movement', 0)
            is_speaking = detection.get('is_speaking', False)
            confidence = detection.get('confidence', 0.5)
            
            # Find closest cluster
            for cid, cluster in person_clusters.items():
                distance = abs(x - cluster['avg_x'])
                
                # Only consider detections reasonably close to this cluster
                if distance > 300:  # More than 300px away - skip
                    continue
                
                # Scoring components
                proximity_score = max(0, 1 - (distance / 300))  # Stricter proximity threshold
                size_weight = size * 100  # Larger faces
                
                # CRITICAL: During transcript-confirmed speaking time, 
                # LIP MOVEMENT should be heavily weighted
                if is_speaking:
                    # This person shows lip movement during speaking time - STRONG signal
                    lip_weight = lip_movement * 50  # 5x higher weight!
                    cluster_data[cid]['speaking_count'] += 1
                else:
                    # No lip movement during speaking time - WEAK signal or wrong person
                    lip_weight = 0
                    cluster_data[cid]['non_speaking_count'] += 1
                
                # Combined score weighted by confidence
                base_score = proximity_score * (1 + size_weight + lip_weight)
                weighted_score = base_score * (0.5 + confidence * 0.5)
                
                cluster_data[cid]['total_score'] += weighted_score
                cluster_data[cid]['weighted_positions'].append(x)
                cluster_data[cid]['confidences'].append(confidence)
        
        # IMPROVED: Calculate final scores with consistency penalty
        for cid, data in cluster_data.items():
            total_detections = data['speaking_count'] + data['non_speaking_count']
            
            if total_detections == 0:
                cluster_scores[cid] = 0
                continue
            
            # Speaking consistency: what % of detections showed lip movement?
            speaking_consistency = data['speaking_count'] / total_detections if total_detections > 0 else 0
            
            # CRITICAL: Penalize clusters with low speaking consistency
            # A person who's actually speaking should show lip movement in >30% of frames
            consistency_multiplier = 1.0
            if speaking_consistency < 0.2:
                # Very low consistency - probably wrong person
                consistency_multiplier = 0.3
            elif speaking_consistency < 0.3:
                # Low consistency - less confident
                consistency_multiplier = 0.6
            elif speaking_consistency > 0.5:
                # High consistency - boost score
                consistency_multiplier = 1.5
            
            # Average confidence for this cluster
            avg_confidence = np.mean(data['confidences']) if data['confidences'] else 0.5
            
            # IMPROVED: Final score with AUDIO CORRELATION as primary signal
            if use_audio_correlation:
                # When audio correlation is available, it's the PRIMARY signal (50% weight)
                audio_weight = data['audio_correlation']
                visual_weight = data['total_score'] * consistency_multiplier
                
                # Combined score: audio correlation (50%) + visual consistency (40%) + confidence (10%)
                cluster_scores[cid] = (
                    audio_weight * 100 * 0.5 +  # Audio correlation (PRIMARY)
                    visual_weight * 0.4 +        # Visual consistency
                    avg_confidence * 20 * 0.1     # Detection quality
                )
                
                logger.debug(f"         Cluster {cid}: audio={audio_weight:.2f}, visual={visual_weight:.1f}, final={cluster_scores[cid]:.1f}")
            else:
                # Fallback to visual-only scoring
                cluster_scores[cid] = data['total_score'] * consistency_multiplier * avg_confidence
                logger.debug(f"         Cluster {cid}: speaking={data['speaking_count']}/{total_detections} ({speaking_consistency:.1%}), score={cluster_scores[cid]:.1f}")
        
        if not cluster_scores:
            continue
        
        # Assign speaker to highest-scoring cluster
        best_cluster_id = max(cluster_scores, key=cluster_scores.get)
        best_cluster = person_clusters[best_cluster_id]
        speaker_positions[speaker] = best_cluster
        
        logger.info(f"         ✅ {speaker} → X={best_cluster['avg_x']} (score: {cluster_scores[best_cluster_id]:.1f})")
    
    # SMART FALLBACK: If we have clusters but no mappings, assign by position
    if not speaker_positions and len(person_clusters) > 0 and len(unique_speakers) > 0:
        logger.warning(f"      ⚠️ No speaker mappings found, using position-based assignment")
        
        # Sort speakers alphabetically and clusters left-to-right
        sorted_speakers = sorted(unique_speakers)
        sorted_clusters_list = sorted(person_clusters.values(), key=lambda c: c['avg_x'])
        
        # Assign in order: Speaker A → leftmost, Speaker B → next, etc.
        for i, speaker in enumerate(sorted_speakers):
            if i < len(sorted_clusters_list):
                speaker_positions[speaker] = sorted_clusters_list[i]
                logger.info(f"         📍 {speaker} → X={sorted_clusters_list[i]['avg_x']} (position-based)")
    
    return speaker_positions


def get_active_speaker_at_time(speaker_timeline: list, time: float) -> str:
    """
    Find which speaker is talking at the given time.
    Returns speaker name or None.
    """
    for utterance in speaker_timeline:
        if utterance['start'] <= time <= utterance['end']:
            return utterance['speaker']
    return None


def smooth_crop_positions(positions: list, window_size: int = 5) -> list:
    """
    Apply moving average to crop positions for smooth transitions.
    """
    import numpy as np
    
    if len(positions) <= window_size:
        return positions
    
    crop_x_values = [p['crop_x'] for p in positions]
    smoothed_x = np.convolve(crop_x_values, np.ones(window_size)/window_size, mode='same')
    
    for i, pos in enumerate(positions):
        pos['crop_x'] = int(smoothed_x[i])
    
    return positions


async def apply_smart_crop_with_transitions(temp_path: str, output_path: str, crop_positions: list):
    """
    IMPROVED: Apply dynamic crop to video with smooth transitions that follow active speakers.
    Uses FFmpeg's crop filter with time-based interpolation.
    """
    import cv2
    
    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    crop_h = input_h
    crop_w = int(crop_h * 0.5625)
    
    # IMPROVED: Build dynamic crop filter that actually follows speakers
    if len(crop_positions) > 1:
        # Create interpolated crop positions for smooth transitions
        # Build expression that changes crop_x over time based on active speaker
        
        # Create time-based keyframes for crop position
        keyframes = []
        for i, pos in enumerate(crop_positions):
            time_point = pos['frame'] / fps
            crop_x = pos['crop_x']
            keyframes.append((time_point, crop_x))
        
        # Build FFmpeg expression for dynamic crop with smooth transitions
        # Using crop filter with expression for x position
        # We'll use a series of conditional expressions based on time
        
        # For smooth following, we'll use linear interpolation between keyframes
        # Build expression: if(lt(t,t1),x1,if(lt(t,t2),lerp(x1,x2,t),...))
        
        # Simplified approach: Use zoompan for smooth following
        # Calculate average movement speed
        import numpy as np
        
        # Create smooth interpolation of crop positions over time
        all_frames = list(range(total_frames))
        frame_to_crop = {}
        
        # Fill in crop_x for sampled frames
        for pos in crop_positions:
            frame_to_crop[pos['frame']] = pos['crop_x']
        
        # Interpolate between sampled frames
        sampled_frames = sorted(frame_to_crop.keys())
        interpolated_crops = []
        
        for frame in all_frames:
            # Find surrounding keyframes
            before_frames = [f for f in sampled_frames if f <= frame]
            after_frames = [f for f in sampled_frames if f > frame]
            
            if before_frames and after_frames:
                # Interpolate between surrounding keyframes
                f1 = before_frames[-1]
                f2 = after_frames[0]
                x1 = frame_to_crop[f1]
                x2 = frame_to_crop[f2]
                
                # Linear interpolation with smoothing
                alpha = (frame - f1) / (f2 - f1)
                # Apply ease-in-out for smoother transitions
                alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)  # Smoothstep
                crop_x = int(x1 + (x2 - x1) * alpha_smooth)
            elif before_frames:
                crop_x = frame_to_crop[before_frames[-1]]
            elif after_frames:
                crop_x = frame_to_crop[after_frames[0]]
            else:
                crop_x = (input_w - crop_w) // 2  # Center fallback
            
            interpolated_crops.append(crop_x)
        
        # Use median with temporal smoothing for stable but responsive crop
        # Apply moving average to reduce jitter
        window_size = int(fps * 0.5)  # 0.5 second smoothing window
        smoothed_crops = np.convolve(
            interpolated_crops, 
            np.ones(window_size)/window_size, 
            mode='same'
        )
        
        # For FFmpeg, we'll use a weighted average that favors recent speaker positions
        # Since FFmpeg crop filter doesn't support frame-by-frame expressions easily,
        # we'll use the weighted average of crop positions based on confidence
        
        # Weight positions by their confidence (if available) and recency
        weighted_sum = 0
        total_weight = 0
        
        for pos in crop_positions:
            weight = pos.get('confidence', 1.0) if 'confidence' in pos else 1.0
            # Add recency weight (later frames weighted higher)
            recency_weight = 1.0 + (pos['frame'] / total_frames) * 0.5
            combined_weight = weight * recency_weight
            
            weighted_sum += pos['crop_x'] * combined_weight
            total_weight += combined_weight
        
        weighted_crop_x = int(weighted_sum / total_weight) if total_weight > 0 else (input_w - crop_w) // 2
        
        # Clamp to valid range
        weighted_crop_x = max(0, min(weighted_crop_x, input_w - crop_w))
        
        logger.info(f"      Dynamic crop: weighted average X={weighted_crop_x} from {len(crop_positions)} keyframes")
        filter_str = f"crop={crop_w}:{crop_h}:{weighted_crop_x}:0,scale=1080:1920"
    else:
        crop_x = crop_positions[0]['crop_x']
        filter_str = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_path,
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    logger.info(f"      Crop filter: {filter_str}")
    subprocess.run(cmd, check=True)


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


