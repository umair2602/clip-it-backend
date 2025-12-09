"""
Audio-Visual Correlation Module

Extracts audio energy from video and correlates it with visual mouth movements
to accurately identify which person is speaking.
"""

import logging
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("librosa or soundfile not available. Audio-visual correlation will use fallback method.")
    AUDIO_LIBS_AVAILABLE = False


def extract_audio_from_video(video_path: str) -> Tuple[np.ndarray, int]:
    """
    Extract audio waveform from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (audio_waveform, sample_rate)
    """
    try:
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio using FFmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate (sufficient for speech)
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            temp_audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if AUDIO_LIBS_AVAILABLE:
            # Load with librosa for better processing
            audio, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
        else:
            # Fallback: load with basic numpy (PCM 16-bit)
            with open(temp_audio_path, 'rb') as f:
                # Skip WAV header (44 bytes)
                f.seek(44)
                audio_bytes = f.read()
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        logger.info(f"   Extracted audio: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f}s)")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        # Return silence as fallback
        return np.zeros(16000), 16000


def calculate_frame_audio_energy(
    audio: np.ndarray,
    sr: int,
    fps: float,
    num_frames: int
) -> np.ndarray:
    """
    Calculate audio energy for each video frame.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        fps: Video frame rate
        num_frames: Number of frames in video
        
    Returns:
        Array of audio energy per frame (length = num_frames)
    """
    # Calculate samples per frame
    samples_per_frame = int(sr / fps)
    
    frame_energies = []
    
    for frame_idx in range(num_frames):
        # Get audio samples for this frame
        start_sample = int(frame_idx * samples_per_frame)
        end_sample = int((frame_idx + 1) * samples_per_frame)
        
        if end_sample > len(audio):
            # Pad with zeros if we run out of audio
            frame_audio = np.zeros(samples_per_frame)
            if start_sample < len(audio):
                available = len(audio) - start_sample
                frame_audio[:available] = audio[start_sample:len(audio)]
        else:
            frame_audio = audio[start_sample:end_sample]
        
        # Calculate RMS energy for this frame
        energy = np.sqrt(np.mean(frame_audio ** 2))
        frame_energies.append(energy)
    
    frame_energies = np.array(frame_energies)
    
    # Normalize to 0-1 range
    if frame_energies.max() > 0:
        frame_energies = frame_energies / frame_energies.max()
    
    logger.info(f"   Calculated audio energy for {num_frames} frames (mean: {frame_energies.mean():.3f})")
    
    return frame_energies


def calculate_mouth_opening(face_landmarks) -> float:
    """
    Calculate mouth opening/aperture from MediaPipe face landmarks.
    
    Uses mouth landmarks to measure vertical opening.
    More accurate than simple lip distance.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        
    Returns:
        Mouth opening value (0-1, normalized)
    """
    # MediaPipe Face Mesh landmark indices for mouth
    # Upper lip: landmarks 13, 14
    # Lower lip: landmarks 78, 308
    # Mouth corners: 61, 291
    
    # Get vertical mouth opening (upper to lower lip)
    upper_lip_indices = [13, 14]
    lower_lip_indices = [78, 308, 87, 317]
    
    upper_y = np.mean([face_landmarks.landmark[i].y for i in upper_lip_indices])
    lower_y = np.mean([face_landmarks.landmark[i].y for i in lower_lip_indices])
    
    # Vertical opening
    vertical_opening = abs(lower_y - upper_y)
    
    # Also measure horizontal opening (mouth corners)
    left_corner = face_landmarks.landmark[61].x
    right_corner = face_landmarks.landmark[291].x
    horizontal_opening = abs(right_corner - left_corner)
    
    # Combine vertical and horizontal (aspect ratio indicates mouth state)
    # Wide + tall opening = speaking
    # Just wide = smile
    # Just tall = surprise
    mouth_opening = vertical_opening * (1 + horizontal_opening)
    
    return float(mouth_opening)


def correlate_audio_visual(
    audio_energies: np.ndarray,
    mouth_openings: List[float],
    frames: List[int]
) -> float:
    """
    Calculate correlation between audio energy and mouth opening.
    
    High correlation = this person is likely speaking
    Low correlation = this person is NOT speaking (just reacting/laughing)
    
    Args:
        audio_energies: Audio energy per frame (full video)
        mouth_openings: Mouth opening measurements for this person
        frames: Frame indices for mouth_openings
        
    Returns:
        Correlation score (0-1, higher = more likely speaking)
    """
    if len(mouth_openings) < 3:
        # Not enough data points
        return 0.0
    
    # Extract audio energies for the frames we have mouth data for
    audio_subset = audio_energies[frames]
    
    # Normalize both signals to 0-1
    mouth_openings_normalized = np.array(mouth_openings)
    if mouth_openings_normalized.max() > 0:
        mouth_openings_normalized = mouth_openings_normalized / mouth_openings_normalized.max()
    
    # Calculate Pearson correlation coefficient
    try:
        correlation = np.corrcoef(audio_subset, mouth_openings_normalized)[0, 1]
        
        # Handle NaN (happens when one signal is constant)
        if np.isnan(correlation):
            correlation = 0.0
        
        # Convert to 0-1 range (correlation is -1 to 1, we only care about positive)
        correlation_score = max(0.0, correlation)
        
        return float(correlation_score)
        
    except Exception as e:
        logger.warning(f"Error calculating correlation: {e}")
        return 0.0


def calculate_time_aligned_correlation(
    audio_energies: np.ndarray,
    person_detections: List[Dict],
    speaking_windows: List[Tuple[float, float]],
    fps: float
) -> float:
    """
    Calculate correlation ONLY during speaking windows (transcript-confirmed times).
    
    This combines:
    - Audio-visual correlation (visual matches audio)
    - Temporal alignment (happens during transcript-confirmed speaking time)
    
    Args:
        audio_energies: Audio energy per frame
        person_detections: List of detection dicts with 'mouth_opening' and 'frame'
        speaking_windows: List of (start_time, end_time) tuples when speaker is talking
        fps: Video frame rate
        
    Returns:
        Weighted correlation score (0-1)
    """
    if not person_detections or not speaking_windows:
        return 0.0
    
    # Filter detections to only those during speaking windows
    speaking_detections = []
    
    for detection in person_detections:
        frame_time = detection['frame'] / fps
        
        # Check if this frame is during a speaking window
        is_speaking_time = any(
            start <= frame_time <= end
            for start, end in speaking_windows
        )
        
        if is_speaking_time:
            speaking_detections.append(detection)
    
    if len(speaking_detections) < 3:
        return 0.0
    
    # Extract mouth openings and frames
    mouth_openings = [d['mouth_opening'] for d in speaking_detections]
    frames = [d['frame'] for d in speaking_detections]
    
    # Calculate correlation
    correlation = correlate_audio_visual(audio_energies, mouth_openings, frames)
    
    # Weight by percentage of frames analyzed (more data = more confident)
    data_coverage = min(1.0, len(speaking_detections) / 10)  # 10+ frames = full confidence
    
    weighted_correlation = correlation * (0.5 + 0.5 * data_coverage)
    
    return weighted_correlation
