"""
YOLO26 Pose-Based Face Detection Service
Uses yolo26n-pose.pt to get real face keypoints (nose, eyes, ears)
instead of guessing face position from a person bounding box.
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
            logger.info(f"🚀 YOLO26 using GPU (CUDA): {torch.cuda.get_device_name(0)}")
        else:
            _device = 'cpu'
            logger.info("💻 YOLO26 using CPU (no CUDA available)")
    return _device


def get_yolo_model():
    """Load and cache YOLO26 pose model for accurate face keypoint detection"""
    global _yolo_model
    
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            
            device = get_device()
            
            # Use YOLO26n-pose: detects persons AND 17 body keypoints
            # Keypoints 0-4 are head: nose, left_eye, right_eye, left_ear, right_ear
            # This gives us REAL face coordinates instead of guessing top-40% of body
            logger.info("📦 Loading YOLO26 pose model for accurate face detection...")
            
            _yolo_model = YOLO('yolo26n-pose.pt')  # Pose nano model
            _yolo_model.to(device)
            
            logger.info(f"✅ YOLO26 pose model loaded on {device}")
            
        except ImportError:
            logger.error("❌ ultralytics not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading YOLO pose model: {str(e)}")
            raise
    
    return _yolo_model


# COCO 17-keypoint indices for the head region
_HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear


def _extract_face_from_keypoints(keypoints, person_idx: int, person_bbox: tuple) -> tuple:
    """
    Extract real face vertical center from YOLO26 pose keypoints.
    Horizontal center always uses the PERSON bbox center (not nose/eye position)
    so the crop window centers the whole person, not just their face.

    COCO keypoint order:
        0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear

    Args:
        keypoints: result.keypoints object from YOLO pose inference
        person_idx: index of this person in the result
        person_bbox: (x, y, w, h) person bounding box

    Returns:
        (face_x, face_y, face_w, face_h, person_center_x, face_center_y)
        NOTE: person_center_x is the BODY horizontal center for crop positioning.
              face_center_y is the HEAD vertical center from keypoints.
    """
    x, y, w, h = person_bbox

    # Horizontal: always use person body center so the crop window
    # frames the whole person, regardless of which way they face.
    person_center_x = int(x + w / 2)
    logger.debug(f"[BBOX] person bbox=({x},{y},{w},{h}) → person_center_x={person_center_x}")

    # Fallback vertical: top 25% of person bbox
    fallback_face_h = int(h * 0.25)
    fallback_cy = int(y + fallback_face_h / 2)
    fallback = (x, y, w, fallback_face_h, person_center_x, fallback_cy)

    if keypoints is None:
        return fallback

    try:
        kp_xy = keypoints.xy[person_idx].cpu().numpy()    # shape (17, 2)
        kp_conf = (
            keypoints.conf[person_idx].cpu().numpy()
            if keypoints.conf is not None
            else np.ones(17)
        )

        # Collect head keypoints with sufficient confidence
        visible_pts = [
            kp_xy[idx]
            for idx in _HEAD_KP_INDICES
            if kp_conf[idx] > 0.3 and kp_xy[idx][0] > 0 and kp_xy[idx][1] > 0
        ]

        if not visible_pts:
            return fallback

        pts = np.array(visible_pts)
        # face_center_y: real vertical position of head from keypoints
        face_center_y = int(np.mean(pts[:, 1]))

        # Estimate face size from keypoint spread + generous padding
        if len(visible_pts) >= 2:
            spread = max(
                np.max(pts[:, 0]) - np.min(pts[:, 0]),
                np.max(pts[:, 1]) - np.min(pts[:, 1])
            )
            face_size = int(spread * 2.8)
        else:
            face_size = int(w * 0.55)  # ~55% of person width

        face_size = max(face_size, 30)  # never smaller than 30px
        half = face_size // 2

        # face_bbox is centered on the HEAD (vertical from keypoints)
        # but crop center_x uses person body center (horizontal)
        logger.debug(f"[KP] nose/eye/ear pts={[list(p) for p in visible_pts[:3]]} → face_cy={face_center_y}, body_cx={person_center_x}")
        return (
            person_center_x - half,
            face_center_y - half,
            face_size,
            face_size,
            person_center_x,   # <- body center X for horizontal crop tracking
            face_center_y,     # <- head center Y from real keypoints
        )

    except Exception:
        return fallback


def detect_faces_yolo(frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Detect faces/persons in a frame using YOLO26 pose model.
    Returns real face center coordinates derived from head keypoints
    (nose, eyes, ears) rather than a heuristic estimate.

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
        results = model(frame, conf=conf_threshold, device=device, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints  # available from pose model

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Get real face position from head keypoints
                face_x, face_y, face_w, face_h, face_center_x, face_center_y = \
                    _extract_face_from_keypoints(keypoints, i, (x, y, w, h))

                detections.append({
                    'bbox': (x, y, w, h),
                    'face_bbox': (face_x, face_y, face_w, face_h),
                    'confidence': confidence,
                    'center_x': face_center_x,
                    'center_y': face_center_y,
                    'face_width': face_w,
                    'face_height': face_h
                })

        return detections

    except Exception as e:
        logger.error(f"Error in YOLO face detection: {str(e)}")
        return []


def detect_faces_batch_yolo(frames: List[np.ndarray], conf_threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
    """
    Batch process multiple frames for better GPU utilization.
    Uses YOLO26 pose model so face center is derived from real head keypoints.

    Args:
        frames: List of BGR images from OpenCV
        conf_threshold: Confidence threshold for detections

    Returns:
        List of detection lists (one per frame)
    """
    model = get_yolo_model()
    device = get_device()

    try:
        results = model(frames, conf=conf_threshold, device=device, verbose=False, stream=False)

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            keypoints = result.keypoints

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                face_x, face_y, face_w, face_h, face_center_x, face_center_y = \
                    _extract_face_from_keypoints(keypoints, i, (x, y, w, h))

                frame_detections.append({
                    'bbox': (x, y, w, h),
                    'face_bbox': (face_x, face_y, face_w, face_h),
                    'confidence': confidence,
                    'center_x': face_center_x,
                    'center_y': face_center_y,
                    'face_width': face_w,
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
