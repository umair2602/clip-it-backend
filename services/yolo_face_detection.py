"""
YOLO8 Face Detection Service
Replaces MediaPipe with YOLOv8 for better performance and GPU support
"""

import logging
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Global variables for model caching
_yolo_model = None
_device = None


def get_device():
    """Determine if GPU is available"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = 'cuda'
            logger.info(f"🚀 YOLO8 using GPU (CUDA): {torch.cuda.get_device_name(0)}")
        else:
            _device = 'cpu'
            logger.info("💻 YOLO8 using CPU (no CUDA available)")
    return _device


def get_yolo_model():
    """Load and cache YOLO8 model for face detection"""
    global _yolo_model
    
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            
            device = get_device()
            
            # Use YOLOv8n (nano) for face detection - fast and accurate
            # You can use a custom trained model or pretrained person detection
            logger.info("📦 Loading YOLOv8 model...")
            
            # Option 1: Use YOLOv8 person detection (class 0 = person)
            # Then crop to face region using upper body heuristics
            _yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
            _yolo_model.to(device)
            
            logger.info(f"✅ YOLOv8 model loaded on {device}")
            
        except ImportError:
            logger.error("❌ ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading YOLO model: {str(e)}")
            raise
    
    return _yolo_model


def detect_faces_yolo(frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Detect faces/persons in a frame using YOLO8.
    
    Args:
        frame: BGR image from OpenCV
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detections with bounding boxes and confidence scores
        [{'bbox': (x, y, w, h), 'confidence': 0.95, 'center_x': 640, 'center_y': 360}, ...]
    """
    model = get_yolo_model()
    device = get_device()
    
    try:
        # Run YOLO inference
        results = model(frame, conf=conf_threshold, device=device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Filter for person class (class 0 in COCO)
                if int(box.cls[0]) == 0:  # Person class
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Calculate bbox parameters
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Estimate face region (upper 40% of person bbox)
                    face_h = int(h * 0.4)
                    face_center_x = int(x + w / 2)
                    face_center_y = int(y + face_h / 2)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'face_bbox': (x, y, w, face_h),  # Estimated face region
                        'confidence': confidence,
                        'center_x': face_center_x,
                        'center_y': face_center_y,
                        'face_width': w,
                        'face_height': face_h
                    })
        
        return detections
        
    except Exception as e:
        logger.error(f"Error in YOLO face detection: {str(e)}")
        return []


def detect_faces_batch_yolo(frames: List[np.ndarray], conf_threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
    """
    Batch process multiple frames for better GPU utilization.
    
    Args:
        frames: List of BGR images from OpenCV
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detection lists (one per frame)
    """
    model = get_yolo_model()
    device = get_device()
    
    try:
        # Batch inference
        results = model(frames, conf=conf_threshold, device=device, verbose=False, stream=False)
        
        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for box in boxes:
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    face_h = int(h * 0.4)
                    face_center_x = int(x + w / 2)
                    face_center_y = int(y + face_h / 2)
                    
                    frame_detections.append({
                        'bbox': (x, y, w, h),
                        'face_bbox': (x, y, w, face_h),
                        'confidence': confidence,
                        'center_x': face_center_x,
                        'center_y': face_center_y,
                        'face_width': w,
                        'face_height': face_h
                    })
            
            all_detections.append(frame_detections)
        
        return all_detections
        
    except Exception as e:
        logger.error(f"Error in YOLO batch face detection: {str(e)}")
        return [[] for _ in frames]


def group_detections_into_tracks(detections: List[Dict[str, Any]], 
                                 max_distance: int = 100) -> List[List[Dict[str, Any]]]:
    """
    Group detections across frames into speaker tracks based on spatial proximity.
    
    Args:
        detections: List of all person detections with frame info
        max_distance: Maximum pixel distance to consider same person
        
    Returns:
        List of tracks (each track is a list of detections for one person)
    """
    if not detections:
        return []
    
    tracks = []
    
    for detection in detections:
        center_x = detection['center_x']
        center_y = detection['center_y']
        
        # Try to assign to existing track
        assigned = False
        for track in tracks:
            # Check if close to any detection in this track
            track_centers = [(d['center_x'], d['center_y']) for d in track]
            avg_x = np.mean([c[0] for c in track_centers])
            avg_y = np.mean([c[1] for c in track_centers])
            
            distance = np.sqrt((center_x - avg_x)**2 + (center_y - avg_y)**2)
            
            if distance < max_distance:
                track.append(detection)
                assigned = True
                break
        
        if not assigned:
            # Create new track
            tracks.append([detection])
    
    # Filter out tracks with too few detections (likely noise)
    tracks = [t for t in tracks if len(t) >= 3]
    
    logger.info(f"📊 Grouped {len(detections)} detections into {len(tracks)} speaker tracks")
    
    return tracks


# Fallback to OpenCV Haar Cascade if YOLO fails (CPU-only environments)
_haar_cascade = None

def get_haar_cascade():
    """Load Haar Cascade for CPU fallback"""
    global _haar_cascade
    if _haar_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _haar_cascade = cv2.CascadeClassifier(cascade_path)
    return _haar_cascade


def detect_faces_haar(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Fallback face detection using OpenCV Haar Cascade (CPU only).
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        List of detections in same format as YOLO
    """
    cascade = get_haar_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    detections = []
    for (x, y, w, h) in faces:
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        detections.append({
            'bbox': (x, y, w, h),
            'face_bbox': (x, y, w, h),
            'confidence': 0.8,  # Haar doesn't provide confidence
            'center_x': center_x,
            'center_y': center_y,
            'face_width': w,
            'face_height': h
        })
    
    return detections


def detect_faces(frame: np.ndarray, use_yolo: bool = True) -> List[Dict[str, Any]]:
    """
    Main face detection function with automatic fallback.
    
    Args:
        frame: BGR image from OpenCV
        use_yolo: Try YOLO first, fallback to Haar if fails
        
    Returns:
        List of face detections
    """
    if use_yolo:
        try:
            return detect_faces_yolo(frame)
        except Exception as e:
            logger.warning(f"YOLO detection failed, falling back to Haar: {str(e)}")
            return detect_faces_haar(frame)
    else:
        return detect_faces_haar(frame)
