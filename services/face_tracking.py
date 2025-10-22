import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Configure MediaPipe for GPU acceleration if available
def get_mediapipe_gpu_config():
    """Configure MediaPipe to use GPU acceleration if available"""
    try:
        # Check if CUDA is available via OpenCV
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if cuda_available:
            print(f"[INFO] CUDA-enabled GPU detected for MediaPipe. Device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
            # Configure MediaPipe to use GPU
            return {
                "model_selection": 1,  # 0 for short-range, 1 for full-range detection
                "min_detection_confidence": 0.5,
                "gpu_origin": True  # Enable GPU acceleration
            }
        else:
            print("[INFO] No CUDA-enabled GPU detected for MediaPipe. Using CPU.")
            return {
                "model_selection": 1,
                "min_detection_confidence": 0.5
            }
    except Exception as e:
        print(f"[WARNING] Error checking GPU availability: {str(e)}. Using CPU for MediaPipe.")
        return {
            "model_selection": 1,
            "min_detection_confidence": 0.5
        }

# Use GPU configuration
MEDIAPIPE_CONFIG = get_mediapipe_gpu_config()

async def track_faces(video_path: str) -> List[Dict[str, Any]]:
    """Track faces in a video using MediaPipe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of face tracking data for each frame
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize face detection
        with mp_face_detection.FaceDetection(**MEDIAPIPE_CONFIG) as face_detection:
            
            # Process frames
            tracking_data = []
            frame_count = 0
            
            # Process every nth frame to speed up processing
            # Adjust the step size based on your performance requirements
            step = max(1, int(fps / 5))  # Process 5 frames per second
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frame_count % step == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the frame
                    results = face_detection.process(frame_rgb)
                    
                    # Extract face data
                    frame_data = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'faces': []
                    }
                    
                    if results.detections:
                        for detection in results.detections:
                            # Get bounding box
                            bbox = detection.location_data.relative_bounding_box
                            
                            # Convert relative coordinates to absolute
                            x = int(bbox.xmin * width)
                            y = int(bbox.ymin * height)
                            w = int(bbox.width * width)
                            h = int(bbox.height * height)
                            
                            # Calculate center point
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # Add face data
                            face_data = {
                                'x': center_x,
                                'y': center_y,
                                'width': w,
                                'height': h,
                                'confidence': detection.score[0]
                            }
                            
                            frame_data['faces'].append(face_data)
                    
                    # Add frame data to tracking data
                    tracking_data.append(frame_data)
                
                frame_count += 1
                
                # Break if we've processed enough frames
                if frame_count >= total_frames:
                    break
            
            # Release the video capture
            cap.release()
            
            # Interpolate missing frames
            if step > 1 and tracking_data:
                tracking_data = interpolate_tracking_data(tracking_data, total_frames, step)
            
            return tracking_data
    
    except Exception as e:
        print(f"Error tracking faces: {str(e)}")
        # Return empty tracking data on error
        return []

def interpolate_tracking_data(tracking_data: List[Dict[str, Any]], total_frames: int, step: int) -> List[Dict[str, Any]]:
    """Interpolate face tracking data for frames that were skipped.
    
    Args:
        tracking_data: List of face tracking data for processed frames
        total_frames: Total number of frames in the video
        step: Step size used for processing
        
    Returns:
        Interpolated tracking data for all frames
    """
    # Create a full list of frames
    full_tracking_data = []
    
    # Iterate through all frames
    for frame in range(total_frames):
        # Find the nearest processed frames
        prev_idx = frame // step
        next_idx = min(prev_idx + 1, len(tracking_data) - 1)
        
        # If we're at a processed frame, use that data
        if frame % step == 0 and prev_idx < len(tracking_data):
            full_tracking_data.append(tracking_data[prev_idx])
            continue
        
        # Otherwise, interpolate between the nearest processed frames
        if prev_idx < len(tracking_data) and next_idx < len(tracking_data):
            prev_data = tracking_data[prev_idx]
            next_data = tracking_data[next_idx]
            
            # Calculate interpolation factor
            factor = (frame % step) / step
            
            # Create interpolated frame data
            frame_data = {
                'frame': frame,
                'timestamp': frame / (total_frames / tracking_data[-1]['timestamp']),
                'faces': []
            }
            
            # Interpolate face data
            if prev_data['faces'] and next_data['faces']:
                # Match faces between frames (simplified - just using the first face)
                if prev_data['faces'] and next_data['faces']:
                    prev_face = prev_data['faces'][0]
                    next_face = next_data['faces'][0]
                    
                    # Interpolate face position and size
                    face_data = {
                        'x': int(prev_face['x'] + factor * (next_face['x'] - prev_face['x'])),
                        'y': int(prev_face['y'] + factor * (next_face['y'] - prev_face['y'])),
                        'width': int(prev_face['width'] + factor * (next_face['width'] - prev_face['width'])),
                        'height': int(prev_face['height'] + factor * (next_face['height'] - prev_face['height'])),
                        'confidence': prev_face['confidence'] + factor * (next_face['confidence'] - prev_face['confidence'])
                    }
                    
                    frame_data['faces'].append(face_data)
            elif prev_data['faces']:
                # If only previous frame has faces, use those
                frame_data['faces'] = prev_data['faces']
            elif next_data['faces']:
                # If only next frame has faces, use those
                frame_data['faces'] = next_data['faces']
            
            full_tracking_data.append(frame_data)
        else:
            # If we can't interpolate, create an empty frame
            full_tracking_data.append({
                'frame': frame,
                'timestamp': frame / (total_frames / tracking_data[-1]['timestamp'] if tracking_data else 1),
                'faces': []
            })
    
    return full_tracking_data

async def get_face_coordinates(video_path: str, frame_number: int) -> Dict[str, Any]:
    """Get face coordinates for a specific frame.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to process
        
    Returns:
        Dictionary with face coordinates
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Seek to the specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number}")
        
        # Initialize face detection
        with mp_face_detection.FaceDetection(**MEDIAPIPE_CONFIG) as face_detection:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_detection.process(frame_rgb)
            
            # Extract face data
            face_data = {
                'frame': frame_number,
                'faces': []
            }
            
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    
                    # Calculate center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Add face data
                    face_data['faces'].append({
                        'x': center_x,
                        'y': center_y,
                        'width': w,
                        'height': h,
                        'confidence': detection.score[0]
                    })
            
            # Release the video capture
            cap.release()
            
            return face_data
    
    except Exception as e:
        print(f"Error getting face coordinates: {str(e)}")
        # Return empty face data on error
        return {'frame': frame_number, 'faces': []}