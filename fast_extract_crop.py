
def extract_and_crop_clip_fast(
    input_path: str, output_path: str, start_time: float, end_time: float
) -> None:
    """Extract a clip and crop to 9:16 vertical format in ONE FFmpeg command.
    This is MUCH faster than extract + crop separately.

    Args:
        input_path: Path to the input video
        output_path: Path to save the output clip
        start_time: Start time of the clip in seconds
        end_time: End time of the clip in seconds
    """
    try:
        # Calculate duration
        duration = end_time - start_time

        # Get input video dimensions
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            input_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split(','))
        
        # Calculate 9:16 crop dimensions (centered)
        target_aspect = 9 / 16
        crop_width = int(height * target_aspect)
        
        # Ensure crop width doesn't exceed input width
        if crop_width > width:
            crop_width = width
        
        # Calculate crop position (center)
        crop_x = (width - crop_width) // 2
        
        logger.info(f"      Input: {width}x{height}, Crop: {crop_width}x{height} from x={crop_x}")

        # Use hardware acceleration if available
        hw_accel_cmd = get_hardware_acceleration_cmd()

        if hw_accel_cmd and "cuda" in " ".join(hw_accel_cmd):
            try:
                # CUDA accelerated: Extract + Crop + Encode in one command
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start_time),  # Fast seek before input
                    "-i", input_path,
                    "-t", str(duration),
                    "-vf", f"crop={crop_width}:{height}:{crop_x}:0",  # Center crop to 9:16
                    "-c:v", "h264_nvenc",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    output_path,
                ]
                
                logger.info(f"      Using CUDA acceleration with crop filter")
                subprocess.run(cmd, check=True, capture_output=True)
                return
            except subprocess.CalledProcessError as e:
                logger.warning(f"      CUDA encoding failed: {e}, falling back to CPU")

        # CPU fallback: Extract + Crop + Encode (ultrafast preset)
       logger.info("      Using CPU encoding with ultrafast preset")
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),  # Fast seek before input
            "-accurate_seek",
            "-i", input_path,
            "-t", str(duration),
            "-vf", f"crop={crop_width}:{height}:{crop_x}:0",  # Center crop to 9:16
            "-c:v", "libx264",
            "-preset", "ultrafast",  # Fastest encoding
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    except Exception as e:
        logger.error(f"Error in fast extract and crop: {str(e)}")
        raise

