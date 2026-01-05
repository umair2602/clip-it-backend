"""
TalkNet Active Speaker Detection Integration

This module provides audio-visual active speaker detection using TalkNet.
TalkNet analyzes video frames + audio to determine which face is currently speaking.

Based on: https://github.com/TaoRuijie/TalkNet-ASD
Paper: "Is Someone Speaking? Exploring Long-term Temporal Features for Audio-visual Active Speaker Detection" (ACM MM 2021)
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# TalkNet dependencies - will be checked at runtime
TALKNET_AVAILABLE = False
TALKNET_PATH = None
_TALKNET_INSTANCE = None  # Global singleton instance to avoid re-initialization

def check_talknet_installation():
    """Check if TalkNet is installed and available."""
    global TALKNET_AVAILABLE, TALKNET_PATH
    
    try:
        import torch
        import cv2
        import python_speech_features
        from scipy.io import wavfile
        
        # Check if TalkNet model files exist
        model_path = os.path.join(os.path.dirname(__file__), 'talknet_model', 'pretrain_TalkSet.model')
        if not os.path.exists(model_path):
            # Try to download the model
            logger.info("TalkNet model not found, attempting to download...")
            download_talknet_model()
        
        if os.path.exists(model_path):
            TALKNET_AVAILABLE = True
            TALKNET_PATH = os.path.dirname(model_path)
            logger.info(f"TalkNet is available at {TALKNET_PATH}")
        else:
            logger.warning("TalkNet model could not be downloaded")
            TALKNET_AVAILABLE = False
            
    except ImportError as e:
        logger.warning(f"TalkNet dependencies not available: {e}")
        TALKNET_AVAILABLE = False
    
    return TALKNET_AVAILABLE


def download_talknet_model():
    """Download the pretrained TalkNet model."""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'talknet_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'pretrain_TalkSet.model')
        
        # Download using gdown (Google Drive)
        # TalkSet model ID: 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea
        try:
            import gdown
            url = "https://drive.google.com/uc?id=1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
            gdown.download(url, model_path, quiet=False)
            logger.info(f"TalkNet model downloaded to {model_path}")
        except Exception as e:
            logger.warning(f"Could not download with gdown: {e}")
            # Try with subprocess
            cmd = f'gdown --id 1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea -O "{model_path}"'
            subprocess.run(cmd, shell=True, check=True)
            
    except Exception as e:
        logger.error(f"Failed to download TalkNet model: {e}")


class TalkNetASD:
    """TalkNet Active Speaker Detection wrapper for integration with video processing."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize TalkNet model.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        import torch
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.loaded = False
        
        logger.info(f"Initializing TalkNet ASD on device: {self.device}")
        
    def load_model(self):
        """Load the pretrained TalkNet model."""
        if self.loaded:
            return
            
        try:
            import torch
            import torch.nn as nn
            
            # Import TalkNet model architecture
            from services.talknet_model.talkNetModel import talkNetModel
            from services.talknet_model.loss import lossAV
            
            self.model = talkNetModel()
            self.lossAV = lossAV()
            
            if self.device == 'cuda':
                self.model = self.model.cuda()
                self.lossAV = self.lossAV.cuda()
            
            # Load pretrained weights
            model_path = os.path.join(os.path.dirname(__file__), 'talknet_model', 'pretrain_TalkSet.model')
            
            if self.device == 'cpu':
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(model_path)
            
            # Load state dict with name mapping
            model_state = self.model.state_dict()
            for name, param in state_dict.items():
                if name.startswith('model.'):
                    name = name[6:]  # Remove 'model.' prefix
                if name in model_state:
                    if model_state[name].size() == param.size():
                        model_state[name].copy_(param)
            
            self.model.eval()
            self.loaded = True
            logger.info("TalkNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TalkNet model: {e}")
            raise
    
    def extract_face_features(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract visual features from a face crop.
        
        Args:
            face_crop: Face image crop (BGR, any size)
            
        Returns:
            Visual features array
        """
        import cv2
        
        # Convert to grayscale and resize to 112x112 (TalkNet input size)
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop
            
        face = cv2.resize(gray, (224, 224))
        # Center crop to 112x112
        face = face[56:168, 56:168]
        
        return face
    
    def extract_audio_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract MFCC features from audio.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate (should be 16000)
            
        Returns:
            MFCC features array
        """
        import python_speech_features
        
        # Ensure audio is a valid numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        mfcc = python_speech_features.mfcc(
            audio, 
            sample_rate, 
            numcep=13, 
            winlen=0.025, 
            winstep=0.010
        )
        
        # Ensure MFCC is a proper 2D array
        if not isinstance(mfcc, np.ndarray):
            mfcc = np.array(mfcc)
        
        if mfcc.ndim == 1:
            mfcc = mfcc.reshape(-1, 1)
        elif mfcc.ndim == 0:
            # Scalar - this shouldn't happen but handle it
            mfcc = np.array([[mfcc]])
        
        return mfcc
    
    def detect_active_speaker(
        self, 
        video_frames: List[np.ndarray],
        face_tracks: List[Dict[str, Any]],
        audio: np.ndarray,
        sample_rate: int = 16000,
        fps: float = 25.0
    ) -> List[Dict[str, float]]:
        """Detect active speaker for each face track.
        
        Args:
            video_frames: List of video frames (BGR)
            face_tracks: List of face track dictionaries with 'frame', 'bbox' keys
            audio: Audio waveform
            sample_rate: Audio sample rate
            fps: Video frame rate
            
        Returns:
            List of dicts with 'track_id', 'frame', 'score' for each face/frame
        """
        import torch
        import cv2
        
        if not self.loaded:
            self.load_model()
        
        # Extract audio features for the entire clip
        logger.info(f"🎵 Extracting MFCC audio features from {len(audio)} samples...")
        audio_features = self.extract_audio_features(audio, sample_rate)
        
        # Log audio features info
        logger.info(f"✅ MFCC features extracted: shape={audio_features.shape}, dtype={audio_features.dtype}")
        
        results = []
        
        # Process each face track
        total_tracks = len(face_tracks)
        logger.info(f"🔄 Processing {total_tracks} face tracks for active speaker detection...")
        for track_idx, track in enumerate(face_tracks):
            if track_idx % 5 == 0 or track_idx == total_tracks - 1:
                logger.info(f"  ⏳ Processing track {track_idx + 1}/{total_tracks}...")
            track_frames = track.get('frames', [])
            track_bboxes = track.get('bboxes', [])
            
            if not track_frames or not track_bboxes:
                continue
            
            # Extract face crops for this track
            video_features = []
            for frame_idx, bbox in zip(track_frames, track_bboxes):
                if frame_idx < len(video_frames):
                    frame = video_frames[frame_idx]
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    
                    # Ensure valid bbox
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2]
                        face_features = self.extract_face_features(face_crop)
                        video_features.append(face_features)
            
            if not video_features:
                continue
            
            video_features = np.array(video_features)
            
            # Align audio and video features
            # Audio: 100 frames/second, Video: fps frames/second
            audio_start = int(track_frames[0] / fps * 100)
            audio_end = int((track_frames[-1] + 1) / fps * 100)
            
            # Ensure valid audio indices
            if audio_start < 0:
                audio_start = 0
            if audio_end > len(audio_features):
                audio_end = len(audio_features)
            if audio_start >= audio_end:
                logger.warning(f"Invalid audio segment indices for track {track_idx} (start={audio_start}, end={audio_end}), skipping")
                continue
            
            audio_segment = audio_features[audio_start:audio_end]
            
            # Ensure we have matching lengths
            video_length = len(video_features)
            
            # Validate audio_segment is a valid numpy array with correct shape
            if audio_segment is None:
                logger.warning(f"None audio segment for track {track_idx}, skipping")
                continue
                
            # Convert to numpy array if it isn't already
            if not isinstance(audio_segment, np.ndarray):
                logger.warning(f"Audio segment is not a numpy array (type={type(audio_segment)}) for track {track_idx}, converting...")
                try:
                    audio_segment = np.array(audio_segment)
                except Exception as conv_error:
                    logger.warning(f"Failed to convert audio segment to array for track {track_idx}: {conv_error}, skipping")
                    continue
            
            # Check if it's a 0-dimensional array (scalar)
            if audio_segment.ndim == 0:
                logger.warning(f"Audio segment is a scalar for track {track_idx}, skipping")
                continue
                
            # Check if empty
            if audio_segment.size == 0:
                logger.warning(f"Empty audio segment for track {track_idx}, skipping")
                continue
            
            # Ensure audio_segment is 2D (T, F) for MFCC
            if audio_segment.ndim == 1:
                audio_segment = audio_segment.reshape(-1, 1)
            
            audio_length = len(audio_segment)
            
            # Calculate expected ratio
            expected_audio = int(video_length * 100 / fps)
            if expected_audio <= 0:
                expected_audio = 1
                
            if audio_length < expected_audio:
                # Pad audio - use correct padding for 2D array (T, F)
                pad_length = expected_audio - audio_length
                
                # CRITICAL FIX: Convert pad_length to Python int (numpy scalars cause .pad() error)
                try:
                    # Force conversion to Python int
                    pad_length = int(pad_length)
                    
                    # Ensure audio_segment is a proper 2D numpy array
                    if not isinstance(audio_segment, np.ndarray):
                        audio_segment = np.array(audio_segment)
                    
                    if audio_segment.ndim != 2:
                        if audio_segment.ndim == 1:
                            audio_segment = audio_segment.reshape(-1, 1)
                        elif audio_segment.ndim == 0:
                            logger.warning(f"Audio segment is scalar for track {track_idx}, skipping")
                            continue
                        else:
                            logger.warning(f"Cannot handle audio_segment with ndim={audio_segment.ndim}, skipping track {track_idx}")
                            continue
                    
                    # Now do the padding with proper tuple format
                    # ((before_rows, after_rows), (before_cols, after_cols))
                    audio_segment = np.pad(audio_segment, ((0, pad_length), (0, 0)), mode='edge')
                    
                except (TypeError, ValueError, AttributeError) as pad_error:
                    logger.error(f"Failed to pad audio for track {track_idx}: {pad_error}")
                    logger.error(f"  pad_length={pad_length} (type={type(pad_length)})")
                    logger.error(f"  audio_segment type={type(audio_segment)}, shape={getattr(audio_segment, 'shape', 'N/A')}")
                    continue
                    
            elif audio_length > expected_audio:
                audio_segment = audio_segment[:expected_audio]
            
            # Run TalkNet inference
            try:
                scores = self._run_inference(video_features, audio_segment, fps)
                
                # Map scores back to frames
                for i, frame_idx in enumerate(track_frames):
                    if i < len(scores):
                        results.append({
                            'track_id': track_idx,
                            'frame': frame_idx,
                            'score': float(scores[i]),
                            'bbox': track_bboxes[i] if i < len(track_bboxes) else None
                        })
            except Exception as e:
                logger.error(f"TalkNet inference failed for track {track_idx}: {e}")
                logger.error(f"  video_features shape: {video_features.shape if hasattr(video_features, 'shape') else type(video_features)}")
                logger.error(f"  audio_segment shape: {audio_segment.shape if hasattr(audio_segment, 'shape') else type(audio_segment)}")
                logger.error(f"  fps: {fps}")
                import traceback
                logger.error(f"  Full traceback: {traceback.format_exc()}")
        
        return results
    
    def _run_inference(
        self, 
        video_features: np.ndarray, 
        audio_features: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """Run TalkNet model inference.
        
        Args:
            video_features: Array of face features (N, 112, 112)
            audio_features: MFCC features array
            fps: Video frame rate
            
        Returns:
            Speaking scores for each frame
        """
        import torch
        
        # Validate inputs
        if video_features is None or len(video_features) == 0:
            return np.array([])
        
        if audio_features is None or len(audio_features) == 0:
            # Return neutral scores if no audio
            return np.full(len(video_features), 0.5)
        
        # Ensure audio_features is 2D (T, F)
        if audio_features.ndim == 1:
            audio_features = audio_features.reshape(-1, 1)
        
        with torch.no_grad():
            # Prepare inputs
            video_tensor = torch.FloatTensor(video_features).unsqueeze(0)
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
            
            if self.device == 'cuda':
                video_tensor = video_tensor.cuda()
                audio_tensor = audio_tensor.cuda()
            
            # Forward pass
            audio_embed = self.model.forward_audio_frontend(audio_tensor)
            video_embed = self.model.forward_visual_frontend(video_tensor)
            
            # Cross-attention
            audio_embed, video_embed = self.model.forward_cross_attention(audio_embed, video_embed)
            
            # Backend
            output = self.model.forward_audio_visual_backend(audio_embed, video_embed)
            
            # Get scores
            scores = self.lossAV.forward(output, labels=None)
            
            if isinstance(scores, tuple):
                scores = scores[0]
            
            scores = scores.cpu().numpy().flatten()
            
        return scores


async def detect_active_speaker_simple(
    video_path: str,
    face_detections: List[Dict[str, Any]],
    fps: float
) -> Dict[int, float]:
    """Simplified active speaker detection for integration with existing pipeline.
    
    This function takes the existing face detections and adds speaking scores.
    
    Args:
        video_path: Path to video file
        face_detections: List of face detection dicts with 'frame', 'x', 'y', 'size' keys
        fps: Video frame rate
        
    Returns:
        Dict mapping frame_number -> {face_x: speaking_score}
    """
    import cv2
    from scipy.io import wavfile
    
    global _TALKNET_INSTANCE
    
    if not TALKNET_AVAILABLE:
        logger.warning("TalkNet not available, falling back to basic detection")
        return {}
    
    try:
        # Use singleton instance to avoid re-initialization
        if _TALKNET_INSTANCE is None:
            logger.info("🎬 Initializing TalkNet model (first time only)...")
            _TALKNET_INSTANCE = TalkNetASD()
            _TALKNET_INSTANCE.load_model()
            logger.info("✅ TalkNet model loaded and cached for reuse")
        else:
            logger.info("♻️ Reusing cached TalkNet model instance")
        
        talknet = _TALKNET_INSTANCE
        
        # Extract audio from video
        audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav')
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path, '-loglevel', 'panic'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Read audio
        sample_rate, audio = wavfile.read(audio_path)
        
        # Validate audio data
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure audio is the right type for processing
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)
        
        # Log audio info for debugging
        logger.info(f"Audio loaded: shape={audio.shape}, dtype={audio.dtype}, sample_rate={sample_rate}")
        
        # Group detections into tracks (by spatial proximity)
        logger.info(f"📊 Grouping {len(face_detections)} face detections into speaker tracks...")
        tracks = _group_detections_into_tracks(face_detections, fps)
        logger.info(f"✅ Found {len(tracks)} speaker tracks to analyze")
        
        # Read video frames
        logger.info(f"🎥 Loading video frames for analysis...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"  ⏳ Loaded {frame_count} frames so far...")
        cap.release()
        logger.info(f"✅ Loaded {len(frames)} frames from video")
        
        # Run TalkNet
        logger.info(f"🔍 Running TalkNet inference on {len(tracks)} tracks across {len(frames)} frames...")
        logger.info(f"⏳ This may take several minutes on CPU (faster with GPU)...")
        results = talknet.detect_active_speaker(frames, tracks, audio, sample_rate, fps)
        logger.info(f"✅ TalkNet analysis complete! Processed {len(results)} frame predictions")
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Convert results to frame -> {x: score} format
        frame_scores = {}
        for r in results:
            frame = r['frame']
            if frame not in frame_scores:
                frame_scores[frame] = {}
            
            if r['bbox']:
                center_x = (r['bbox'][0] + r['bbox'][2]) / 2
                frame_scores[frame][int(center_x)] = r['score']
        
        return frame_scores
        
    except Exception as e:
        logger.error(f"TalkNet detection failed: {e}")
        return {}


def _group_detections_into_tracks(
    detections: List[Dict[str, Any]], 
    fps: float,
    distance_threshold: float = 100
) -> List[Dict[str, Any]]:
    """Group face detections into tracks based on spatial proximity.
    
    Args:
        detections: List of detection dicts
        fps: Video frame rate
        distance_threshold: Max distance between consecutive detections
        
    Returns:
        List of track dicts with 'frames' and 'bboxes' keys
    """
    import numpy as np
    
    if not detections:
        return []
    
    # Sort by frame
    sorted_dets = sorted(detections, key=lambda d: d.get('frame', 0))
    
    tracks = []
    used = set()
    
    for i, det in enumerate(sorted_dets):
        if i in used:
            continue
        
        # Start a new track
        track = {
            'frames': [det.get('frame', 0)],
            'bboxes': [],
            'centers': [det.get('x', 0)]
        }
        
        # Estimate bbox from center and size
        x = det.get('x', 0)
        y = det.get('y', 0)
        size = det.get('size', 0.1) * 1000  # Approximate size
        half_size = size / 2
        
        track['bboxes'].append([x - half_size, y - half_size, x + half_size, y + half_size])
        
        used.add(i)
        last_x = x
        last_frame = det.get('frame', 0)
        
        # Find connected detections
        for j, other_det in enumerate(sorted_dets[i+1:], start=i+1):
            if j in used:
                continue
            
            other_frame = other_det.get('frame', 0)
            other_x = other_det.get('x', 0)
            
            # Check if close enough (spatially and temporally)
            frame_gap = other_frame - last_frame
            if frame_gap > fps:  # More than 1 second gap
                break
            
            distance = abs(other_x - last_x)
            if distance < distance_threshold:
                track['frames'].append(other_frame)
                
                # Estimate bbox
                ox = other_det.get('x', 0)
                oy = other_det.get('y', 0)
                osize = other_det.get('size', 0.1) * 1000
                ohalf = osize / 2
                track['bboxes'].append([ox - ohalf, oy - ohalf, ox + ohalf, oy + ohalf])
                track['centers'].append(ox)
                
                used.add(j)
                last_x = ox
                last_frame = other_frame
        
        if len(track['frames']) >= 5:  # Minimum track length
            tracks.append(track)
    
    return tracks


# Check availability on module load
check_talknet_installation()
