import os
import logging
import asyncio
from pathlib import Path
import sieve
import shutil
import tempfile
from config import settings
from typing import Optional, Callable

# Initialize Sieve with API key
sieve.api_key = settings.SIEVE_API_KEY

# Check if API key is configured
if not sieve.api_key or sieve.api_key == "":
    logging.warning("⚠️  SIEVE_API_KEY is not configured - Sieve downloader will fail")
else:
    logging.info(f"✅ Sieve API key configured (length: {len(sieve.api_key)} chars)")

# Global cancellation flag for the current download
_cancellation_flag = {"cancelled": False, "user_id": None, "video_id": None}

def set_download_cancellation_context(user_id: str, video_id: str):
    """Set the cancellation context for the download"""
    _cancellation_flag["cancelled"] = False
    _cancellation_flag["user_id"] = user_id
    _cancellation_flag["video_id"] = video_id
    logging.info(f"🔒 Download cancellation context set: user={user_id}, video={video_id}")

def clear_download_cancellation_context():
    """Clear the cancellation context"""
    _cancellation_flag["cancelled"] = False
    _cancellation_flag["user_id"] = None
    _cancellation_flag["video_id"] = None

def mark_download_cancelled():
    """Mark the current download as cancelled"""
    _cancellation_flag["cancelled"] = True
    logging.warning(f"🛑 Download marked as cancelled")

def is_download_cancelled() -> bool:
    """Check if the current download is cancelled"""
    return _cancellation_flag.get("cancelled", False)

async def download_youtube_video_sieve(url: str, output_dir: Path, user_id: str = None, video_id: str = None) -> tuple:
    """
    Download a YouTube video using Sieve API
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        tuple: (file_path, title, video_info)
    """
    try:
        # Early validation - check if API key is set
        if not sieve.api_key or sieve.api_key == "":
            logging.error("❌ SIEVE_API_KEY is not configured - skipping Sieve download")
            return None, None, None
        
        logging.info(f"🎬 Starting Sieve YouTube download")
        logging.info(f"   URL: {url}")
        logging.info(f"   Output dir: {output_dir}")
        logging.info(f"   Output dir exists: {os.path.exists(output_dir)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"✅ Output directory ready: {output_dir}")
        
        # Run the download in a separate thread to avoid blocking
        logging.info(f"🔄 Running download in executor thread...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _download_video_sieve(url, output_dir))
        
        logging.info(f"📊 Executor completed")
        logging.info(f"   Result: {result}")
        
        return result
    
    except Exception as e:
        logging.error(f"❌ Error in download_youtube_video_sieve wrapper")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        logging.error(f"   URL: {url}")
        logging.error(f"   Output dir: {output_dir}")
        logging.exception("Full stack trace:")
        return None, None, None

def _download_video_sieve(url: str, output_dir: Path) -> tuple:
    """
    Internal function to download YouTube video using Sieve API (runs in executor)
    """
    try:
        logging.info(f"🔧 Initializing Sieve YouTube downloader...")
        logging.info(f"   Sieve API key set: {bool(sieve.api_key)}")
        logging.info(f"   Sieve API key length: {len(sieve.api_key) if sieve.api_key else 0}")
        
        # Initialize Sieve YouTube downloader
        youtube_downloader = sieve.function.get("sieve/youtube-downloader")
        logging.info(f"✅ Sieve downloader initialized: {type(youtube_downloader)}")
        
        # Set parameters for the Sieve API
        download_type = "video"
        resolution = "highest-available"
        include_audio = True
        start_time = 0
        end_time = -1
        include_metadata = True
        # Use only valid metadata fields as per the error message
        metadata_fields = ["title", "thumbnail", "duration", "channel_id", "view_count", "upload_date"]
        include_subtitles = False
        subtitle_languages = []
        video_format = "mp4"
        audio_format = "mp3"
        subtitle_format = "vtt"
        
        logging.info(f"📋 Sieve API parameters:")
        logging.info(f"   download_type: {download_type}")
        logging.info(f"   resolution: {resolution}")
        logging.info(f"   include_audio: {include_audio}")
        logging.info(f"   video_format: {video_format}")
        logging.info(f"   metadata_fields: {metadata_fields}")
        
        # Call the Sieve API to download the video
        logging.info(f"🚀 Calling Sieve API for URL: {url}")
        output = youtube_downloader.run(
            url=url,
            download_type=download_type,
            resolution=resolution,
            include_audio=include_audio,
            start_time=start_time,
            end_time=end_time,
            include_metadata=include_metadata,
            metadata_fields=metadata_fields,
            include_subtitles=include_subtitles,
            subtitle_languages=subtitle_languages,
            video_format=video_format,
            audio_format=audio_format,
            subtitle_format=subtitle_format
        )
        logging.info(f"✅ Sieve API call completed")
        logging.info(f"   Output type: {type(output)}")
        
        # Process the output - Sieve returns 2 objects: metadata dict + file
        logging.info(f"🔄 Starting to iterate over Sieve output generator...")
        logging.info(f"   This may take several minutes while Sieve processes the video remotely...")
        output_count = 0
        metadata = {}
        temp_file_path = None
        
        for output_object in output:
            # Check for cancellation before processing each output
            if is_download_cancelled():
                logging.warning(f"🛑 Download cancelled during Sieve processing")
                raise Exception("Download cancelled by user")
            
            output_count += 1
            logging.info(f"🎯 Received output object from Sieve generator (iteration {output_count})")
            logging.info(f"📦 Processing Sieve output object #{output_count}")
            logging.info(f"   Object type: {type(output_object)}")
            logging.info(f"   Object content: {output_object}")
            
            # First object is usually metadata (dict)
            if isinstance(output_object, dict):
                logging.info(f"   📋 Found metadata dictionary")
                metadata = output_object
                logging.info(f"   Metadata keys: {list(metadata.keys())}")
                continue
            
            # Second object is the file (sieve.File)
            # Get the file path from the output
            if hasattr(output_object, 'path'):
                # Check cancellation before file download
                if is_download_cancelled():
                    logging.warning(f"🛑 Download cancelled before file retrieval")
                    raise Exception("Download cancelled by user")
                    
                logging.info(f"   ⏳ Accessing .path property (this triggers download from Sieve)...")
                temp_file_path = output_object.path
                logging.info(f"   ✅ Found file path: {temp_file_path}")
                break  # We have the file, exit loop
            else:
                logging.warning(f"   ⚠️ Object has no 'path' attribute")
                logging.warning(f"   Object attributes: {dir(output_object)}")
                continue
                
        if not temp_file_path:
            logging.error("❌ No file path found in any output object")
            logging.error(f"   Total objects received: {output_count}")
            logging.error(f"   Metadata: {metadata}")
            return None, None, None
                
        logging.info(f"📂 Downloaded file from Sieve: {temp_file_path}")
        logging.info(f"   File exists: {os.path.exists(temp_file_path) if temp_file_path else False}")
        
        # Get or generate title from metadata
        title = metadata.get('title', f"youtube_{os.path.basename(temp_file_path)}")
        logging.info(f"📝 Video title: {title}")
        
        # Get duration from metadata - it's in "HH:MM:SS" format
        duration_str = metadata.get('duration', '0')
        if isinstance(duration_str, str) and ':' in duration_str:
            # Parse "HH:MM:SS" to seconds
            parts = duration_str.split(':')
            if len(parts) == 3:
                duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                duration = 0
        else:
            duration = metadata.get('duration', 0) if isinstance(metadata.get('duration'), (int, float)) else 0
            
        # Get thumbnail_url from nested dict structure: thumbnails -> high/medium/default -> url
        thumbnail_url = None
        thumbnails = metadata.get('thumbnails', {})
        if isinstance(thumbnails, dict):
            # Try high quality first, then medium, then default
            for quality in ['high', 'medium', 'default']:
                if quality in thumbnails and isinstance(thumbnails[quality], dict):
                    thumbnail_url = thumbnails[quality].get('url')
                    if thumbnail_url:
                        break
        
        logging.info(f"📊 Extracted metadata:")
        logging.info(f"   Duration: {duration}s (from '{duration_str}')")
        logging.info(f"   Thumbnail URL: {thumbnail_url}")
        
        # Create final file path in the output directory
        file_name = f"{title.replace(' ', '_')}.mp4"
        safe_file_name = ''.join(c for c in file_name if c.isalnum() or c in ['_', '-', '.']).rstrip()
        final_file_path = os.path.join(output_dir, safe_file_name)
        logging.info(f"💾 Copying to final location: {final_file_path}")
        
        # Copy the file to the output directory
        shutil.copy2(temp_file_path, final_file_path)
        
        # Verify the copy
        if os.path.exists(final_file_path):
            final_size = os.path.getsize(final_file_path)
            logging.info(f"✅ File copied successfully")
            logging.info(f"   Size: {final_size:,} bytes ({final_size/1024/1024:.2f} MB)")
        else:
            logging.error(f"❌ File copy failed - file does not exist at {final_file_path}")
        
        # Create video info dictionary with the correct field names
        video_info = {
            "title": title,
            "author": metadata.get("channel_id", "Unknown"),
            "length_seconds": duration,
            "thumbnail_url": thumbnail_url or "",
            "resolution": "unknown"
        }
        
        logging.info(f"✅ Sieve download completed successfully")
        logging.info(f"   Final path: {final_file_path}")
        logging.info(f"   Title: {title}")
        logging.info(f"   Video info: {video_info}")
        return final_file_path, title, video_info
            
        # If we get here, no output was returned
        logging.error("❌ No output returned from Sieve API")
        logging.error(f"   Total output objects processed: {output_count}")
        return None, None, None
        
    except Exception as e:
        logging.error(f"❌ Error in _download_video_sieve")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        logging.error(f"   URL: {url}")
        logging.error(f"   Output dir: {output_dir}")
        logging.exception("Full stack trace:")
        return None, None, None 