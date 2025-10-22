import os
import logging
import asyncio
from pathlib import Path
import sieve
import shutil
import tempfile
from config import settings

# Initialize Sieve with API key
sieve.api_key = settings.SIEVE_API_KEY

async def download_youtube_video_sieve(url: str, output_dir: Path) -> tuple:
    """
    Download a YouTube video using Sieve API
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        tuple: (file_path, title, video_info)
    """
    try:
        logging.info(f"Downloading YouTube video via Sieve API: {url}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the download in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _download_video_sieve(url, output_dir))
    
    except Exception as e:
        logging.error(f"Error downloading YouTube video with Sieve: {str(e)}", exc_info=True)
        return None, None, None

def _download_video_sieve(url: str, output_dir: Path) -> tuple:
    """
    Internal function to download YouTube video using Sieve API (runs in executor)
    """
    try:
        # Initialize Sieve YouTube downloader
        youtube_downloader = sieve.function.get("sieve/youtube-downloader")
        
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
        
        # Call the Sieve API to download the video
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
        
        # Process the output
        for output_object in output:
            # Log the output_object to understand its structure
            logging.info(f"Sieve output object type: {type(output_object)}")
            logging.info(f"Sieve output object content: {output_object}")
            
            # Get the file path from the output - using dictionary access instead of attribute access
            if isinstance(output_object, dict):
                temp_file_path = output_object.get('path')
            else:
                # Try attribute access if it's not a dictionary
                temp_file_path = getattr(output_object, 'path', None)
                
            if not temp_file_path:
                logging.error("No path found in output object")
                continue
                
            logging.info(f"Downloaded file from Sieve: {temp_file_path}")
            
            # Extract metadata if available
            metadata = {}
            if isinstance(output_object, dict):
                metadata = output_object.get('metadata', {})
            else:
                metadata = getattr(output_object, 'metadata', {})
                
            if isinstance(metadata, dict) and not metadata:
                # If metadata is empty, check if it's directly in the output_object
                if isinstance(output_object, dict) and 'title' in output_object:
                    metadata = output_object
            
            # Get or generate title
            title = metadata.get('title', f"youtube_{os.path.basename(temp_file_path)}")
            
            # Create final file path in the output directory
            file_name = f"{title.replace(' ', '_')}.mp4"
            safe_file_name = ''.join(c for c in file_name if c.isalnum() or c in ['_', '-', '.']).rstrip()
            final_file_path = os.path.join(output_dir, safe_file_name)
            
            # Copy the file to the output directory
            shutil.copy2(temp_file_path, final_file_path)
            
            # Create video info dictionary with the correct field names
            video_info = {
                "title": title,
                "author": metadata.get("channel_id", "Unknown"),  # Use channel_id instead of author
                "length_seconds": metadata.get("duration", 0),    # Use duration instead of length_seconds
                "thumbnail_url": metadata.get("thumbnail", ""),   # Use thumbnail instead of thumbnail_url
                "resolution": "unknown"  # Sieve might not provide this information
            }
            
            logging.info(f"Download completed: {final_file_path}")
            return final_file_path, title, video_info
            
        # If we get here, no output was returned
        logging.error("No output returned from Sieve API")
        return None, None, None
        
    except Exception as e:
        logging.error(f"Error in _download_video_sieve: {str(e)}", exc_info=True)
        return None, None, None 