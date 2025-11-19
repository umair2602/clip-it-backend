"""
YouTube video downloader using yt-dlp (more reliable than pytubefix)
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import subprocess
import json

logging.basicConfig(level=logging.INFO)

async def download_youtube_video_ytdlp(
    url: str, 
    output_dir: str = "./uploads",
    on_progress=None
) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Download a YouTube video using yt-dlp (subprocess call)
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        on_progress: Optional callback for progress updates (not used with yt-dlp)
    
    Returns:
        Tuple of (file_path, title, video_info)
    """
    try:
        logging.info(f"🎬 Starting yt-dlp download for: {url}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Output template for filename
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
        
        # Get video info first
        info_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-warnings",
            url
        ]
        
        logging.info("📊 Fetching video metadata with yt-dlp...")
        info_result = subprocess.run(
            info_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if info_result.returncode != 0:
            logging.error(f"yt-dlp info fetch failed: {info_result.stderr}")
            return None, None, None
        
        # Parse video info
        video_info_raw = json.loads(info_result.stdout)
        video_id = video_info_raw.get("id")
        title = video_info_raw.get("title", "Unknown Title")
        duration = video_info_raw.get("duration", 0)
        thumbnail = video_info_raw.get("thumbnail", "")
        uploader = video_info_raw.get("uploader", "Unknown")
        
        logging.info(f"📹 Video: {title}")
        logging.info(f"⏱️  Duration: {duration}s")
        logging.info(f"👤 Uploader: {uploader}")
        
        # Download the video
        download_cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",  # Prefer MP4, fallback to best
            "-o", output_template,
            "--no-warnings",
            "--progress",
            url
        ]
        
        logging.info("⬇️  Downloading video with yt-dlp...")
        download_result = subprocess.run(
            download_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if download_result.returncode != 0:
            logging.error(f"yt-dlp download failed: {download_result.stderr}")
            return None, None, None
        
        # Construct the expected file path
        # yt-dlp saves as video_id.ext
        possible_extensions = ["mp4", "webm", "mkv", "m4a"]
        file_path = None
        
        for ext in possible_extensions:
            potential_path = os.path.join(output_dir, f"{video_id}.{ext}")
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            logging.error(f"Downloaded file not found in {output_dir}")
            return None, None, None
        
        # Verify file exists and has size
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logging.error(f"Downloaded file is missing or empty: {file_path}")
            return None, None, None
        
        # Create standardized video_info dict
        video_info = {
            "title": title,
            "author": uploader,
            "length_seconds": duration,
            "thumbnail_url": thumbnail,
            "video_id": video_id,
            "resolution": video_info_raw.get("resolution", "unknown")
        }
        
        logging.info(f"✅ yt-dlp download completed successfully")
        logging.info(f"   File: {file_path}")
        logging.info(f"   Size: {os.path.getsize(file_path)} bytes")
        
        return file_path, title, video_info
        
    except subprocess.TimeoutExpired:
        logging.error("yt-dlp download timed out (5 minutes)")
        return None, None, None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse yt-dlp JSON output: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"❌ Error in yt-dlp download")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        logging.exception("Full stack trace:")
        return None, None, None
