"""
TalkNet-only crop detection - lightweight alternative to MediaPipe
Uses OpenCV for fast face detection + TalkNet for active speaker detection
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Import TalkNet
try:
    from services.talknet_asd import (
        TalkNetASD,
        detect_active_speaker_simple,
        TALKNET_AVAILABLE
    )
except ImportError:
    TALKNET_AVAILABLE = False
    logger.warning("TalkNet not available")

# Constants
MIN_SPEAKER_DURATION = 1.0
SPEAKER_HYSTERESIS = 150


async def detect_talknet_crop_positions(
    video_path: str,
    start_time: float, 
    end_time: float
) -> list:
    """
    Use TalkNet + basic OpenCV face detection to find active speaker and generate crop positions.
    MUCH faster than MediaPipe - uses lightweight face detection + TalkNet audio-visual model.
    
    Strategy:
    1. Use OpenCV Haar Cascade for fast face detection (no MediaPipe)
    2. Run TalkNet on detected faces to find who is speaking
    3. Track the active speaker's position
    4. Generate crop positions to follow the speaker
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"      Video: {input_w}x{input_h} @ {fps:.2f}fps, {total_frames} frames")
    
    # Calculate crop dimensions for 9:16
    crop_h = input_h
    crop_w = int(crop_h * 0.9)  # Tighter crop
    
    if crop_w > input_w:
        crop_w = input_w
    
    logger.info(f"      ✂️  Crop dimensions: {crop_w}x{crop_h}")
    
    # STEP 1: Fast face detection using OpenCV (no MediaPipe)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    person_detections = []
    sample_interval = max(1, int(fps / 10))  # Sample 10 fps
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        
        frame_time = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            face_size = (w * h) / (input_w * input_h)
            
            person_detections.append({
                'frame': frame_idx,
                'time': frame_time,
                'x': center_x,
                'y': center_y,
                'size': face_size,
                'bbox': [x, y, x+w, y+h],
                'is_speaking': False,  # Will be updated by TalkNet
                'confidence': 0.5,
                'type': 'opencv_haar'
            })
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"      ✅ Detected {len(person_detections)} face instances (OpenCV)")
    
    if len(person_detections) == 0:
        logger.warning(f"      ⚠️ No faces detected - using center crop")
        return [{'frame': 0, 'crop_x': (input_w - crop_w) // 2, 'center_x': input_w // 2, 'has_detection': False}]
    
    # STEP 2: Run TalkNet to identify who is speaking
    try:
        if TALKNET_AVAILABLE:
            logger.info(f"      🎙️ Running TalkNet for speaker detection...")
            talknet_scores = await detect_active_speaker_simple(video_path, person_detections, fps)
            
            if talknet_scores:
                # Update detections with TalkNet scores
                talknet_updates = 0
                for det in person_detections:
                    frame = det['frame']
                    x = det['x']
                    
                    if frame in talknet_scores:
                        frame_scores = talknet_scores[frame]
                        best_score = 0.0
                        for face_x, score in frame_scores.items():
                            if abs(face_x - x) < 150:  # Within 150 pixels
                                best_score = max(best_score, score)
                        
                        if best_score > 0:
                            det['is_speaking'] = best_score > 0.5
                            det['talknet_score'] = best_score
                            det['confidence'] = best_score
                            talknet_updates += 1
                
                logger.info(f"      ✅ TalkNet updated {talknet_updates} detections")
            else:
                logger.info(f"      ℹ️ TalkNet returned no scores, using largest face")
        else:
            logger.info(f"      ℹ️ TalkNet not available, using largest face")
    except Exception as e:
        logger.warning(f"      ⚠️ TalkNet failed: {e}, using largest face")
    
    # STEP 3: Generate crop positions following the active speaker
    crop_positions = []
    sampled_frames = sorted(set(d['frame'] for d in person_detections))
    
    current_center_x = None
    last_switch_time = 0
    
    for frame_num in sampled_frames:
        frame_time = frame_num / fps
        frame_detections = [d for d in person_detections if d['frame'] == frame_num]
        
        if not frame_detections:
            center_x = current_center_x if current_center_x is not None else input_w // 2
        else:
            # Find the best speaker
            speaking_detections = [d for d in frame_detections if d.get('is_speaking', False)]
            
            if speaking_detections:
                # Someone is speaking - follow them
                best_speaker = max(speaking_detections, key=lambda d: d.get('talknet_score', 0) * 100 + d.get('size', 0) * 50)
                target_x = best_speaker['x']
            else:
                # No one speaking - follow largest face
                largest = max(frame_detections, key=lambda d: d.get('size', 0))
                target_x = largest['x']
            
            # Apply hysteresis
            if current_center_x is not None:
                distance = abs(target_x - current_center_x)
                time_since_switch = frame_time - last_switch_time
                
                if distance < SPEAKER_HYSTERESIS or time_since_switch < MIN_SPEAKER_DURATION:
                    center_x = current_center_x
                else:
                    center_x = target_x
                    current_center_x = center_x
                    last_switch_time = frame_time
            else:
                center_x = target_x
                current_center_x = center_x
                last_switch_time = frame_time
        
        # Calculate crop position
        crop_x = center_x - (crop_w // 2)
        crop_x = max(0, min(crop_x, input_w - crop_w))
        
        crop_positions.append({
            'frame': frame_num,
            'crop_x': crop_x,
            'center_x': center_x,
            'has_detection': len(frame_detections) > 0
        })
    
    unique_positions = len(set(p['crop_x'] for p in crop_positions))
    logger.info(f"      ✅ Generated {len(crop_positions)} crop positions ({unique_positions} unique)")
    
    return crop_positions
