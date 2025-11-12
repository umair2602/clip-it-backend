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

# Import other services
from services.face_tracking import get_face_coordinates, track_faces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Set up logging
logger = logging.getLogger(__name__)

async def process_video(
    video_path: str, transcript: Dict[str, Any], segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process video to create clips for each engaging segment.

    Args:
        video_path: Path to the video file
        transcript: Transcript from Whisper
        segments: List of engaging segments with start and end times

    Returns:
        List of processed clips with paths
    """
    try:
        logger.info("="*70)
        logger.info(f"📹 STARTING VIDEO CLIP CREATION")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Number of segments to process: {len(segments)}")
        logger.info("="*70)
        
        # Create temporary output directory for processing
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="clip_processing_")
        output_dir = Path(temp_dir)
        logger.info(f"Created temporary output directory: {output_dir}")

        logger.info("\n" + "="*70)
        logger.info("SEGMENTS TO PROCESS:")
        for i, seg in enumerate(segments):
            duration = seg.get('end_time', 0) - seg.get('start_time', 0)
            logger.info(f"  Segment {i+1}: {seg.get('title', 'Untitled')} ({seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s, duration: {duration:.1f}s)")
        logger.info("="*70 + "\n")

        # Process each segment
        processed_clips = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
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
            )

            # Only add to processed clips if a clip was created (faces were detected)
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

                processed_clips.append(
                    {
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
                )

                segment_elapsed = time.time() - segment_start_time
                logger.info(f"   ⏱️  Clip {i+1}/{total_segments} completed in {segment_elapsed:.2f}s")
                logger.info(f"   ✅ Successfully processed: {segment.get('title', 'Untitled')}")
            else:
                logger.warning(f"   ❌ Clip {i+1}/{total_segments} SKIPPED - No faces detected or creation failed")
                logger.warning(f"   Segment: {segment.get('title', 'Untitled')}")

        logger.info("\n" + "="*70)
        logger.info(f"✅ VIDEO PROCESSING COMPLETE")
        logger.info(f"Total clips created: {len(processed_clips)} out of {total_segments} segments")
        logger.info("="*70 + "\n")

        return processed_clips

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        raise


async def create_clip(
    video_path: str, output_dir: str, start_time: float, end_time: float, clip_id: str
) -> str:
    """
    Extract the video segment between start_time and end_time, and convert it to vertical (portrait, 9:16) format with black bars (letterboxing) on top and bottom, preserving the full width of the original video.
    """


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_id}.mp4")

    # Temporary path for the extracted segment
    temp_path = os.path.join(output_dir, f"{clip_id}_temp.mp4")

    # Step 1: Extract the segment
    extract_clip(video_path, temp_path, start_time, end_time)

    # Step 2: Convert to vertical with black bars (letterboxing)
    # Target portrait resolution
    target_width = 1080
    target_height = 1920
    # FFmpeg filter: scale to fit, then pad
    filter_str = (
        f"scale=iw*min({target_width}/iw\,{target_height}/ih):ih*min({target_width}/iw\,{target_height}/ih),"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_path,
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]
    print(f"Running FFmpeg for vertical letterbox: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Remove the temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path


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


def find_continuous_face_segments(
    face_tracking_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find continuous segments where faces are detected.

    Args:
        face_tracking_data: List of face tracking data for each frame

    Returns:
        List of segments with start and end frames where faces are continuously detected
    """
    segments = []
    current_segment = None

    # Minimum number of consecutive frames with faces to consider a valid segment
    min_segment_length = 15  # Approximately half a second at 30fps

    for i, frame_data in enumerate(face_tracking_data):
        has_face = bool(frame_data.get("faces"))

        if has_face:
            # Start a new segment or continue the current one
            if current_segment is None:
                current_segment = {"start_frame": i, "end_frame": i}
            else:
                current_segment["end_frame"] = i
        elif current_segment is not None:
            # End of a segment, check if it's long enough
            segment_length = (
                current_segment["end_frame"] - current_segment["start_frame"] + 1
            )
            if segment_length >= min_segment_length:
                segments.append(current_segment)
            current_segment = None

    # Don't forget the last segment if it's still open
    if current_segment is not None:
        segment_length = (
            current_segment["end_frame"] - current_segment["start_frame"] + 1
        )
        if segment_length >= min_segment_length:
            segments.append(current_segment)

    return segments


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
                    str(start_time),
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
                    "-strict",
                    "experimental",
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
                        str(start_time),
                        "-i",
                        input_path,
                        "-t",
                        str(duration),
                        "-c:v",
                        "h264_nvenc",  # Use NVENC encoder with CUDA
                        "-c:a",
                        "aac",  # Use AAC for audio
                        "-strict",
                        "experimental",
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
                            *hw_accel_cmd,  # Place hardware acceleration before input
                            "-ss",
                            str(start_time),
                            "-i",
                            input_path,
                            "-t",
                            str(duration),
                            "-c:v",
                            "h264_nvenc",  # Use NVENC encoder with CUDA
                            "-c:a",
                            "aac",  # Use AAC for audio
                            "-strict",
                            "experimental",
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
        # Fall back to software encoding with seeking optimization
        print("Using software encoding for extraction")
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-ss",
            str(start_time),
            "-i",
            input_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",  # Software encoding
            "-preset",
            "fast",  # Use faster preset
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_path,
        ]

        print(f"Running fallback FFmpeg command: {' '.join(cmd)}")
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
                str(start_time),
                "-i",
                input_path,
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",  # Fastest preset, lowest quality
                "-c:a",
                "aac",
                "-strict",
                "experimental",
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


def convert_to_vertical(
    input_path: str, output_path: str, face_tracking_data: List[Dict[str, Any]]
) -> None:
    """Convert video to vertical format with face tracking.

    Args:
        input_path: Path to the input video
        output_path: Path to save the output clip
        face_tracking_data: Face tracking data for each frame
    """
    try:
        # Get video properties
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate target dimensions (9:16 aspect ratio)
        target_width = int(height * 9 / 16)
        # Ensure even width for video encoder compatibility (libx264 requires even dimensions)
        if target_width % 2 != 0:
            target_width -= 1

        # Determine if we need to crop or pad
        if target_width <= width:
            # Need to crop horizontally
            crop_width = target_width
            crop_height = height
            pad_width = 0
            pad_height = 0
        else:
            # Need to pad horizontally
            crop_width = width
            crop_height = height
            pad_width = target_width - width
            pad_height = 0

        # Get hardware acceleration command
        hw_accel_cmd = get_hardware_acceleration_cmd()

        if face_tracking_data and any(
            frame.get("faces") for frame in face_tracking_data
        ):
            # Use face tracking to determine crop position
            print("Using face tracking data for dynamic cropping")

            # Create a temporary file for the cropped video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = temp_file.name

            # Extract face positions for each frame
            face_positions = []
            for frame in face_tracking_data:
                faces = frame.get("faces", [])
                if faces:
                    # Use the average x-coordinate of all faces
                    avg_x = sum(face.get("x", 0.5) for face in faces) / len(faces)
                    face_positions.append(avg_x)
                else:
                    # Default to center if no faces detected
                    face_positions.append(0.5)

            # Smooth the face positions to avoid jerky movements
            smoothed_positions = smooth_positions(face_positions, window_size=15)

            # SIMPLIFIED: Use average face position instead of frame-by-frame tracking
            # Calculate average face position to avoid complex FFmpeg filters
            if smoothed_positions:
                avg_face_position = sum(smoothed_positions) / len(smoothed_positions)
            else:
                avg_face_position = 0.5  # Center fallback

            # Calculate single crop position based on average
            max_x_offset = max(0, width - crop_width)
            center_x = int(avg_face_position * width)
            x_offset = max(0, min(center_x - crop_width // 2, max_x_offset))

            print(
                f"Smart crop: avg face position {avg_face_position:.2f}, crop offset: {x_offset}"
            )

            # Use simple crop filter instead of complex frame-by-frame filter
            filter_complex = (
                f"[0:v]crop={crop_width}:{crop_height}:{x_offset}:0[cropped]"
            )

            try:
                # Apply the dynamic cropping
                cmd = [
                    "ffmpeg",
                    "-y",
                    *hw_accel_cmd,  # Hardware acceleration before input
                    "-i",
                    input_path,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[cropped]",
                    "-map",
                    "0:a",
                    "-c:a",
                    "copy",
                    temp_path,
                ]

                print(f"Running dynamic cropping: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

                # Now convert the cropped video to vertical format
                cmd = [
                    "ffmpeg",
                    "-y",
                    *hw_accel_cmd,  # Hardware acceleration before input
                    "-i",
                    temp_path,
                    "-vf",
                    f"scale={target_width}:{height}",
                    "-c:a",
                    "copy",
                    output_path,
                ]

                print(f"Running vertical conversion: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error with hardware acceleration: {e}")
                print("Falling back to software encoding")

                # Fallback to software encoding
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[cropped]",
                    "-map",
                    "0:a",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "copy",
                    temp_path,
                ]

                print(f"Running fallback dynamic cropping: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

                # Convert to vertical format with software encoding
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_path,
                    "-vf",
                    f"scale={target_width}:{height}",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "copy",
                    output_path,
                ]

                print(f"Running fallback vertical conversion: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

            # Clean up temporary file
            os.unlink(temp_path)

        else:
            # No face tracking data, use center crop
            print("No face tracking data available, using center crop")

            # Calculate crop position (center crop) with bounds checking
            max_x_offset = max(0, width - crop_width)
            max_y_offset = max(0, height - crop_height)
            x_offset = min(max_x_offset, (width - crop_width) // 2)
            y_offset = min(max_y_offset, (height - crop_height) // 2)

            # Ensure crop dimensions don't exceed video dimensions
            actual_crop_width = min(crop_width, width)
            actual_crop_height = min(crop_height, height)

            # Ensure even dimensions for video encoder compatibility
            if actual_crop_width % 2 != 0:
                actual_crop_width -= 1
            if actual_crop_height % 2 != 0:
                actual_crop_height -= 1

            print(f"Video dimensions: {width}x{height}")
            print(
                f"Crop settings: {actual_crop_width}x{actual_crop_height} at ({x_offset},{y_offset})"
            )

            try:
                # Apply the center crop and convert to vertical format
                # Check if we have CUDA acceleration
                if hw_accel_cmd and "cuda" in " ".join(hw_accel_cmd):
                    # When using CUDA acceleration, just use hardware decoding but do processing in CPU
                    # This avoids format conversion issues between CUDA memory and filters
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-hwaccel",
                        "cuda",  # Use CUDA for decoding only, don't set -hwaccel_output_format cuda
                        "-i",
                        input_path,
                        "-vf",
                        f"crop={actual_crop_width}:{actual_crop_height}:{x_offset}:{y_offset},scale={target_width}:{height}",
                        "-c:v",
                        "h264_nvenc",  # Use NVENC encoder
                        "-preset",
                        "p1",  # Lower latency preset
                        "-c:a",
                        "copy",
                        output_path,
                    ]
                else:
                    # Without hardware acceleration - use safer parameters
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_path,
                        "-vf",
                        f"crop={actual_crop_width}:{actual_crop_height}:{x_offset}:{y_offset},scale={target_width}:{height}",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-crf",
                        "23",  # Good quality
                        "-pix_fmt",
                        "yuv420p",  # Ensure compatible pixel format
                        "-c:a",
                        "copy",
                        output_path,
                    ]

                print(f"Running center crop and vertical conversion: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error with hardware acceleration: {e}")
                print("Falling back to software encoding")

                # Fallback to software encoding
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-vf",
                    f"crop={actual_crop_width}:{actual_crop_height}:{x_offset}:{y_offset},scale={target_width}:{height}",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "copy",
                    output_path,
                ]

                print(
                    f"Running fallback center crop and vertical conversion: {' '.join(cmd)}"
                )
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


def smooth_positions(positions, window_size=15):
    """Smooth an array of positions using a moving average to prevent jerky camera movement.

    Args:
        positions: List of position values (typically between 0 and 1)
        window_size: Size of the smoothing window

    Returns:
        List of smoothed positions
    """
    if not positions:
        return []

    # Pad the beginning and end with the first and last values
    padded = (
        [positions[0]] * (window_size // 2)
        + positions
        + [positions[-1]] * (window_size // 2)
    )

    # Apply moving average
    smoothed = []
    for i in range(len(positions)):
        window = padded[i : i + window_size]
        smoothed.append(sum(window) / len(window))

    return smoothed
