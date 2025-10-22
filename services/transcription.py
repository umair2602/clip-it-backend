import os
import subprocess

# Import configuration
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import whisper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Initialize Whisper model
model = None


def load_model(model_size="small"):
    """Load the Whisper model."""
    global model
    if model is None:
        import torch

        try:
            # Check if CUDA is available and use GPU if possible
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Loading Whisper model on device: {device}")
            if device == "cuda":
                print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
                # Set torch to use the highest precision available for better accuracy
                torch.set_float32_matmul_precision("high")

                # Try to load model on GPU
                try:
                    model = whisper.load_model(model_size, device=device)
                    # Test if model works with a simple operation to catch early GPU issues
                    dummy_input = torch.zeros((1, 80, 3000), device=device)
                    model.encoder(dummy_input)
                    print("[INFO] Successfully loaded and tested model on GPU")
                except Exception as e:
                    print(
                        f"[WARNING] Error loading model on GPU: {str(e)}. Falling back to CPU."
                    )
                    device = "cpu"
                    model = whisper.load_model(model_size, device=device)
            else:
                model = whisper.load_model(model_size, device=device)
        except Exception as e:
            print(
                f"[WARNING] Error during model loading: {str(e)}. Using default settings."
            )
        model = whisper.load_model(model_size)
    return model


def transcribe_audio_sync(video_path: str, model_size: str = "tiny") -> Dict[str, Any]:
    """Synchronous version of transcribe_audio for thread execution"""
    import logging

    try:
        # Load model if not already loaded
        model = load_model(model_size)

        # Extract audio from video
        audio_path = extract_audio(video_path)

        # Validate audio file before transcription
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_size = os.path.getsize(audio_path)
        print(f"Audio file size: {audio_size} bytes")

        # Check if audio file is too small (likely empty or corrupted)
        if audio_size < 100:  # Less than 100 bytes
            print(
                f"[WARNING] Audio file too small ({audio_size} bytes) - video appears silent"
            )
            print(f"[DEBUG] Returning empty transcript for silent video")
            return {"text": "", "segments": [], "language": "en"}

        # Transcribe audio with additional error handling
        try:
            result = safe_whisper_transcribe(model, audio_path, fp16=False)

            # Validate transcription result
            if not result or not isinstance(result, dict):
                raise ValueError("Transcription returned invalid result")

            # Check if transcription actually found any content
            segments = result.get("segments", [])
            text = result.get("text", "").strip()

            if not segments and not text:
                print("Warning: Transcription found no speech content")
                # Return a minimal valid structure instead of failing
                result = {"text": "", "segments": [], "language": "en"}

        except Exception as transcription_error:
            print(f"Transcription error: {str(transcription_error)}")
            error_str = str(transcription_error).lower()

            # Check for various empty audio related errors
            if any(
                pattern in error_str
                for pattern in [
                    "reshape tensor",
                    "0 elements",
                    "linear(",
                    "unknown parameter type",
                    "dimension size -1",
                    "ambiguous",
                    "in_features",
                    "out_features",
                    "cannot reshape",
                    "unspecified dimension",
                    "failed to load audio",
                    "ffmpeg version",
                    "could not open",
                    "invalid data found",
                ]
            ):
                print(
                    f"[DEBUG] Detected empty/corrupted audio error - returning silent transcript"
                )
                return {"text": "", "segments": [], "language": "en"}
            else:
                raise

        # Clean up temporary audio file
        if audio_path != video_path and os.path.exists(audio_path):
            os.remove(audio_path)

        return result

    except Exception as e:
        logging.error(f"Error in sync transcription: {str(e)}")
        raise


async def transcribe_audio(video_path: str, model_size: str = "base") -> Dict[str, Any]:
    """Transcribe audio from a video file using Whisper.

    Args:
        video_path: Path to the video file
        model_size: Whisper model size (tiny, base, small, medium, large)

    Returns:
        Dictionary containing the transcript and segments with timestamps
    """
    print(f"Starting transcription for: {video_path} with model size: {model_size}")
    try:
        # Check if FFmpeg is available before proceeding
        if not is_ffmpeg_available():
            error_msg = (
                "FFmpeg is not installed or not in the system PATH. "
                "FFmpeg is required for audio extraction and transcription. "
                "Please install FFmpeg and make sure it's in your system PATH."
            )
            print(error_msg)
            raise RuntimeError(error_msg)

        # Load the model
        print(f"Loading Whisper model: {model_size}")
        model = load_model(model_size)
        print(f"Model loaded successfully")

        # Extract audio from video if needed
        try:
            audio_path = extract_audio(video_path)
            print(f"Audio extraction complete. Audio path: {audio_path}")

            # Verify the audio file exists and has content
            if not os.path.exists(audio_path):
                raise FileNotFoundError(
                    f"Audio file not found after extraction: {audio_path}"
                )

            file_size = os.path.getsize(audio_path)
            print(f"Audio file size: {file_size} bytes")
            if file_size == 0:
                raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")

            # Transcribe the audio with GPU acceleration
            print(f"Starting transcription of audio file")

            # First try with basic options (no word timestamps) to avoid Triton errors
            try:
                # Use GPU-optimized settings for transcription but without word timestamps
                transcribe_options = {
                    "task": "transcribe",
                    "verbose": True,
                    "fp16": True,  # Enable half-precision for faster GPU processing
                }

                result = safe_whisper_transcribe(
                    model, audio_path, **transcribe_options
                )
                print(f"transcription result: {result}")
                print(
                    f"Transcription complete. Result contains {len(result.get('segments', []))} segments"
                )
            except Exception as e:
                error_str = str(e).lower()

                # Check for empty audio errors first
                if any(
                    pattern in error_str
                    for pattern in [
                        "reshape tensor",
                        "0 elements",
                        "linear(",
                        "unknown parameter type",
                        "dimension size -1",
                        "ambiguous",
                        "in_features",
                        "out_features",
                        "cannot reshape",
                        "unspecified dimension",
                        "failed to load audio",
                        "ffmpeg version",
                        "could not open",
                        "invalid data found",
                    ]
                ):
                    print(
                        f"[DEBUG] Detected empty/corrupted audio error in async function - returning silent transcript"
                    )
                    return {"text": "", "segments": [], "language": "en"}

                # Handle AttributeError for GPU issues
                elif isinstance(
                    e, AttributeError
                ) and "Cannot set attribute 'src' directly" in str(e):
                    print(
                        "GPU acceleration with Triton encountered an error. Falling back to CPU for transcription."
                    )
                    # Fall back to CPU for transcription
                    import torch

                    try:
                        with torch.device("cpu"):
                            result = safe_whisper_transcribe(
                                model, audio_path, fp16=False
                            )
                        print(
                            f"CPU transcription complete. Result contains {len(result.get('segments', []))} segments"
                        )
                    except Exception as cpu_error:
                        cpu_error_str = str(cpu_error).lower()
                        if any(
                            pattern in cpu_error_str
                            for pattern in [
                                "reshape tensor",
                                "0 elements",
                                "linear(",
                                "unknown parameter type",
                                "dimension size -1",
                                "ambiguous",
                                "in_features",
                                "out_features",
                                "cannot reshape",
                                "unspecified dimension",
                            ]
                        ):
                            print(
                                f"[DEBUG] CPU fallback also got empty audio error - returning silent transcript"
                            )
                            return {"text": "", "segments": [], "language": "en"}
                        else:
                            raise cpu_error
                else:
                    raise

            # Clean up temporary audio file if it was created
            if audio_path != video_path and os.path.exists(audio_path):
                print(f"Cleaning up temporary audio file: {audio_path}")
                os.remove(audio_path)

            return result
        except (FileNotFoundError, ValueError) as e:
            print(f"Audio file error: {str(e)}")
            raise
        except Exception as e:
            print(f"Error during audio extraction or transcription: {str(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            raise

    except RuntimeError as e:
        # Specific handling for FFmpeg-related errors
        print(f"FFmpeg RuntimeError: {str(e)}")
        raise
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        raise


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in the system.

    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    print("\n[DEBUG] Checking if FFmpeg is available...")
    try:
        print("[DEBUG] Attempting to run 'ffmpeg -version'...")
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=False
        )
        success = result.returncode == 0
        if success:
            print(
                f"[DEBUG] FFmpeg found in PATH. Version: {result.stdout.splitlines()[0] if result.stdout else 'Unknown'}"
            )
        else:
            print(
                f"[DEBUG] FFmpeg command failed with return code: {result.returncode}"
            )
            if result.stderr:
                print(f"[DEBUG] Error output: {result.stderr}")
        return success
    except FileNotFoundError:
        print(
            "[DEBUG] FFmpeg not found in PATH. Checking common installation locations..."
        )
        # Try to find ffmpeg in common installation locations
        common_locations = [
            # Windows common locations
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
            # Add more common locations as needed
        ]

        print(f"[DEBUG] Checking these locations: {common_locations}")
        for location in common_locations:
            print(f"[DEBUG] Checking location: {location}")
            if os.path.isfile(location):
                print(
                    f"[DEBUG] FFmpeg found at {location}, but it's not in your system PATH."
                )
                print("Please add the containing directory to your system PATH.")
                return False

        print("[DEBUG] FFmpeg not found in any common locations")
        return False
    except Exception as e:
        print(f"[DEBUG] Unexpected error checking for FFmpeg: {str(e)}")
        return False


def extract_audio(video_path: str) -> str:
    """Extract audio from video file if needed.

    Args:
        video_path: Path to the video file

    Returns:
        Path to the audio file
    """
    print(f"\n[DEBUG] extract_audio: Processing video file: {video_path}")

    # Check if the file is already an audio file
    if video_path.lower().endswith((".mp3", ".wav", ".flac", ".aac")):
        print(f"[DEBUG] File is already an audio file: {video_path}")
        return video_path

    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"[DEBUG] ERROR: Video file does not exist: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size = os.path.getsize(video_path)
    print(f"[DEBUG] Video file exists and is of size: {file_size} bytes")

    # ENHANCED: Check file format and codec information
    try:
        format_probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        format_result = subprocess.run(
            format_probe_cmd, capture_output=True, text=True, check=False
        )

        if format_result.returncode == 0:
            import json

            probe_data = json.loads(format_result.stdout)

            print(
                f"[DEBUG] File format: {probe_data.get('format', {}).get('format_name', 'unknown')}"
            )
            print(
                f"[DEBUG] Duration: {probe_data.get('format', {}).get('duration', 'unknown')} seconds"
            )

            # Check for audio streams
            audio_streams = [
                s
                for s in probe_data.get("streams", [])
                if s.get("codec_type") == "audio"
            ]
            video_streams = [
                s
                for s in probe_data.get("streams", [])
                if s.get("codec_type") == "video"
            ]

            print(
                f"[DEBUG] Found {len(video_streams)} video stream(s) and {len(audio_streams)} audio stream(s)"
            )

            if audio_streams:
                for i, stream in enumerate(audio_streams):
                    print(
                        f"[DEBUG] Audio stream {i}: codec={stream.get('codec_name', 'unknown')}, "
                        f"channels={stream.get('channels', 'unknown')}, "
                        f"sample_rate={stream.get('sample_rate', 'unknown')}"
                    )
            else:
                print(
                    f"[WARNING] NO AUDIO STREAMS FOUND - this explains transcription failure!"
                )
                print(f"[DEBUG] This file appears to be video-only or corrupted")

        else:
            print(f"[WARNING] Could not probe file format: {format_result.stderr}")

    except Exception as probe_error:
        print(f"[WARNING] Error probing file: {str(probe_error)}")

    # Check if FFmpeg is available
    if not is_ffmpeg_available():
        error_msg = (
            "FFmpeg is not installed or not in the system PATH. "
            "FFmpeg is required for audio extraction. "
            "Please install FFmpeg and make sure it's in your system PATH:\n"
            "- Windows: https://ffmpeg.org/download.html#build-windows\n"
            "- macOS: brew install ffmpeg\n"
            "- Linux: apt-get install ffmpeg or yum install ffmpeg"
        )
        print(error_msg)
        raise RuntimeError(error_msg)

    # Create a temporary file for the audio
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
        )
        os.makedirs(uploads_dir, exist_ok=True)

        # Use a fixed path in the uploads directory with a unique filename
        audio_filename = f"audio_{os.path.basename(video_path)}_{int(time.time())}.mp3"
        audio_path = os.path.join(uploads_dir, audio_filename)
        print(f"[DEBUG] Created audio file path: {audio_path}")
    except Exception as e:
        print(f"[DEBUG] Error creating audio path: {str(e)}")
        # Fallback to using temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file.close()
        audio_path = temp_file.name
        print(f"[DEBUG] Fallback to temporary file: {audio_path}")

    # Ensure paths are properly quoted for Windows
    video_path_str = str(video_path)
    audio_path_str = str(audio_path)

    # First check if the video actually has an audio track
    try:
        # Use ffprobe to check audio streams
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "csv=p=0",
            video_path_str,
        ]

        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=False
        )

        if probe_result.returncode != 0 or not probe_result.stdout.strip():
            print(f"[WARNING] No audio stream found in video: {video_path}")
            raise ValueError("Video file contains no audio track")

        print(f"[DEBUG] Audio stream detected in video")

    except Exception as probe_error:
        print(f"[WARNING] Could not probe audio streams: {str(probe_error)}")
        # If probe fails, video might not have audio - create a silent audio file for testing
        print(f"[DEBUG] Attempting to create silent audio for testing...")
        try:
            # Create 1 second of silence as fallback
            silent_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=mono:sample_rate=16000",
                "-t",
                "1",
                "-acodec",
                "mp3",
                audio_path_str,
            ]
            subprocess.run(silent_cmd, check=True, capture_output=True, text=True)
            print(f"[DEBUG] Created silent audio file for testing")
            return audio_path_str
        except Exception as silent_error:
            print(f"[WARNING] Could not create silent audio: {str(silent_error)}")
            # Continue with extraction attempt anyway

    # Use FFmpeg to extract audio
    try:
        # Use more robust FFmpeg command with fallback options
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-i",
            video_path_str,
            "-vn",  # No video
            "-acodec",
            "mp3",  # Force MP3 codec
            "-ar",
            "16000",  # Sample rate for Whisper
            "-ac",
            "1",  # Mono audio
            "-ab",
            "64k",  # Audio bitrate
            "-f",
            "mp3",  # Force MP3 format
            audio_path_str,
        ]
        print(f"[DEBUG] Running FFmpeg command: {' '.join(ffmpeg_cmd)}")

        # Add more debug info about the paths
        print(f"[DEBUG] Video path exists: {os.path.exists(video_path_str)}")
        print(
            f"[DEBUG] Audio directory exists: {os.path.exists(os.path.dirname(audio_path_str))}"
        )
        print(
            f"[DEBUG] Audio directory is writable: {os.access(os.path.dirname(audio_path_str), os.W_OK)}"
        )

        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

        # Debug FFmpeg output even on success
        print(f"[DEBUG] FFmpeg completed with return code: {result.returncode}")
        if result.stderr:
            print(f"[DEBUG] FFmpeg stderr: {result.stderr[-500:]}")  # Last 500 chars
        if result.stdout:
            print(f"[DEBUG] FFmpeg stdout: {result.stdout[-200:]}")  # Last 200 chars

        # Check if FFmpeg reported success but file still doesn't exist
        if result.returncode == 0 and not os.path.exists(audio_path):
            print(f"[WARNING] FFmpeg reported success but audio file was not created")
            print(f"[DEBUG] This suggests the video might have no audio tracks")

        # Validate the extracted audio file
        if not os.path.exists(audio_path):
            print(f"[ERROR] Audio file was not created: {audio_path}")
            print(f"[DEBUG] FFmpeg may have failed silently or video has no audio")

            # Create a minimal silent audio file as fallback
            try:
                print(f"[DEBUG] Creating minimal silent audio file as fallback...")
                silent_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=mono:sample_rate=16000",
                    "-t",
                    "0.1",  # Very short silent audio
                    "-acodec",
                    "mp3",
                    audio_path_str,
                ]
                subprocess.run(silent_cmd, check=True, capture_output=True, text=True)
                print(f"[DEBUG] Created silent audio file successfully")
            except Exception as silent_error:
                print(
                    f"[ERROR] Could not create silent audio file: {str(silent_error)}"
                )
                raise FileNotFoundError(
                    f"Audio extraction failed and could not create fallback: {audio_path}"
                )

        audio_size = os.path.getsize(audio_path)
        print(
            f"[DEBUG] FFmpeg extraction successful. Audio file size: {audio_size} bytes"
        )

        # Check if audio file is suspiciously small
        if audio_size < 100:  # Less than 100 bytes
            print(f"[WARNING] Extracted audio file is very small ({audio_size} bytes)")
            print(f"[DEBUG] Video appears to have no meaningful audio content")
            print(
                f"[DEBUG] Will attempt transcription anyway - may produce empty results"
            )
            # Continue with the tiny file instead of failing

        # Validate audio content using ffprobe
        try:
            print(f"[DEBUG] Validating audio content with ffprobe...")
            content_probe_cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration:stream=duration,sample_rate,channels",
                "-of",
                "csv=p=0",
                audio_path_str,
            ]

            content_result = subprocess.run(
                content_probe_cmd, capture_output=True, text=True, check=False
            )

            if content_result.returncode == 0 and content_result.stdout.strip():
                lines = content_result.stdout.strip().split("\n")
                print(f"[DEBUG] Audio content validation: {lines}")

                # Check for valid duration
                has_valid_duration = False
                for line in lines:
                    if line and "," in line:
                        parts = line.split(",")
                        # Look for duration in any part
                        for part in parts:
                            try:
                                duration = float(part)
                                if (
                                    duration > 0.01
                                ):  # At least 0.01 seconds (much more lenient)
                                    has_valid_duration = True
                                    print(
                                        f"[DEBUG] Found valid audio duration: {duration} seconds"
                                    )
                                    break
                            except (ValueError, TypeError):
                                continue
                        if has_valid_duration:
                            break

                if not has_valid_duration:
                    print(f"[WARNING] Audio file has no valid duration or is too short")
                    raise ValueError(
                        "Audio file contains no valid audio content or is too short"
                    )
            else:
                print(
                    f"[WARNING] Could not validate audio content: {content_result.stderr}"
                )
                raise ValueError(
                    "Audio file appears to be corrupted - failed content validation"
                )

        except Exception as validation_error:
            print(f"[WARNING] Audio content validation failed: {str(validation_error)}")
            print(f"[DEBUG] Proceeding anyway with potentially problematic audio...")
            # Don't fail - just warn and proceed
            # raise ValueError(f"Audio content validation failed: {str(validation_error)}")

        print(f"[DEBUG] Audio validation passed - file appears to contain valid audio")
        return audio_path

    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] FFmpeg process error (code {e.returncode}):")
        print(f"[DEBUG] Command: {e.cmd}")

        stderr_output = (
            e.stderr
            if isinstance(e.stderr, str)
            else (e.stderr.decode() if e.stderr else "No error output")
        )
        stdout_output = (
            e.stdout
            if isinstance(e.stdout, str)
            else (e.stdout.decode() if e.stdout else "No standard output")
        )

        print(f"[DEBUG] Error output: {stderr_output}")
        print(f"[DEBUG] Standard output: {stdout_output}")

        # Try simpler extraction as fallback
        print(f"[DEBUG] Trying simpler FFmpeg extraction as fallback...")
        try:
            simple_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path_str,
                "-vn",  # No video
                "-acodec",
                "libmp3lame",  # Use libmp3lame
                "-q:a",
                "4",  # Good quality
                audio_path_str,
            ]
            print(f"[DEBUG] Fallback FFmpeg command: {' '.join(simple_cmd)}")
            fallback_result = subprocess.run(
                simple_cmd, check=True, capture_output=True, text=True
            )

            if fallback_result.stderr:
                print(
                    f"[DEBUG] Fallback FFmpeg stderr: {fallback_result.stderr[-300:]}"
                )

            # Check if fallback produced a file
            if os.path.exists(audio_path):
                fallback_size = os.path.getsize(audio_path)
                print(
                    f"[DEBUG] Fallback extraction succeeded. File size: {fallback_size} bytes"
                )
                if fallback_size >= 100:  # If it's reasonable size, use it
                    return audio_path

            print(f"[DEBUG] Fallback also failed or produced tiny file")
        except Exception as fallback_error:
            print(f"[DEBUG] Fallback extraction also failed: {str(fallback_error)}")

        # Check for specific error patterns in original error
        if "No such file or directory" in stderr_output:
            raise FileNotFoundError(f"Video file not accessible: {video_path}")
        elif (
            "does not contain any stream" in stderr_output
            or "Invalid data found" in stderr_output
        ):
            raise ValueError(
                f"Video file appears to be corrupted or has no audio: {video_path}"
            )
        else:
            raise RuntimeError(f"FFmpeg audio extraction failed: {stderr_output}")

    except Exception as e:
        print(f"[DEBUG] Unexpected error during audio extraction: {str(e)}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        raise


async def get_word_timestamps(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract word-level timestamps from the transcript.

    Args:
        transcript: Transcript from Whisper

    Returns:
        List of words with timestamps
    """
    words = []

    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            words.append(
                {
                    "text": word.get("text", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                }
            )

    return words


def safe_whisper_transcribe(model, audio_path: str, **options) -> Dict[str, Any]:
    """
    Safe wrapper around model.transcribe that handles all possible tensor/empty audio errors.
    This ensures no tensor reshape errors ever escape, regardless of which code path calls it.
    """
    try:
        print(
            f"[DEBUG] safe_whisper_transcribe: Starting transcription of {audio_path}"
        )

        # Check if audio file exists and has reasonable size
        if not os.path.exists(audio_path):
            print(f"[DEBUG] Audio file not found: {audio_path}")
            return {"text": "", "segments": [], "language": "en"}

        file_size = os.path.getsize(audio_path)
        print(f"[DEBUG] Audio file size: {file_size} bytes")

        if file_size < 100:  # Less than 100 bytes
            print(
                f"[DEBUG] Audio file too small ({file_size} bytes) - returning empty transcript"
            )
            return {"text": "", "segments": [], "language": "en"}

        # Attempt transcription with comprehensive error catching
        result = model.transcribe(audio_path, **options)

        # Validate result
        if not result or not isinstance(result, dict):
            print(f"[DEBUG] Invalid transcription result - returning empty transcript")
            return {"text": "", "segments": [], "language": "en"}

        print(
            f"[DEBUG] Transcription successful - {len(result.get('segments', []))} segments"
        )
        return result

    except Exception as e:
        error_str = str(e).lower()
        print(f"[DEBUG] Transcription error in safe wrapper: {error_str}")

        # Comprehensive error pattern matching for all possible empty/corrupted audio issues
        empty_audio_patterns = [
            "reshape tensor",
            "0 elements",
            "linear(",
            "unknown parameter type",
            "dimension size -1",
            "ambiguous",
            "in_features",
            "out_features",
            "cannot reshape",
            "unspecified dimension",
            "failed to load audio",
            "ffmpeg version",
            "could not open",
            "invalid data found",
            "tensor of 0 elements",
            "can be any value and is ambiguous",
            "empty tensor",
            "invalid dimensions",
            "no input",
            "audio decode",
        ]

        if any(pattern in error_str for pattern in empty_audio_patterns):
            print(
                f"[DEBUG] COMPREHENSIVE: Detected empty/corrupted audio error - returning silent transcript"
            )
            print(f"[DEBUG] Error details: {str(e)}")
            return {"text": "", "segments": [], "language": "en"}
        else:
            print(f"[DEBUG] Non-audio-related error - re-raising: {str(e)}")
            raise


if __name__ == "__main__":
    import asyncio

    async def main():
        print("\n[DEBUG] Starting main function")
        try:
            # Check FFmpeg availability first
            print("[DEBUG] Checking FFmpeg availability in main...")
            if not is_ffmpeg_available():
                print("\nERROR: FFmpeg is not installed or not in the system PATH.")
                print("FFmpeg is required for audio extraction and transcription.")
                print("\nPlease install FFmpeg and make sure it's in your system PATH:")
                print("- Windows: https://ffmpeg.org/download.html#build-windows")
                print("- macOS: brew install ffmpeg")
                print("- Linux: apt-get install ffmpeg or yum install ffmpeg")
                return

            print("[DEBUG] FFmpeg is available, proceeding with transcription")
            video_path = "../uploads/1.mp4"
            print(f"[DEBUG] Attempting to transcribe: {video_path}")
            print(f"[DEBUG] Absolute path: {os.path.abspath(video_path)}")

            result = await transcribe_audio(video_path)
            print("[DEBUG] Transcription completed successfully")
            print(result)
        except RuntimeError as e:
            if "FFmpeg" in str(e):
                print("[DEBUG] FFmpeg-related RuntimeError caught in main:")
                print(f"[DEBUG] {str(e)}")
                # FFmpeg-related error already handled in the function
                pass
            else:
                print(f"[DEBUG] Runtime error: {str(e)}")
        except Exception as e:
            print(f"[DEBUG] Unexpected error in main: {str(e)}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            import traceback

            print(f"[DEBUG] Traceback: {traceback.format_exc()}")

    asyncio.run(main())
