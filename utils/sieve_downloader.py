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
        
        # Process the output
        output_count = 0
        for output_object in output:
            output_count += 1
            # Log the output_object to understand its structure
            logging.info(f"📦 Processing Sieve output object #{output_count}")
            logging.info(f"   Object type: {type(output_object)}")
            logging.info(f"   Object content: {output_object}")
            
            # Get the file path from the output - using dictionary access instead of attribute access
            if isinstance(output_object, dict):
                temp_file_path = output_object.get('path')
                logging.info(f"   Dict access - path: {temp_file_path}")
            else:
                # Try attribute access if it's not a dictionary
                temp_file_path = getattr(output_object, 'path', None)
                logging.info(f"   Attribute access - path: {temp_file_path}")
                
            if not temp_file_path:
                logging.error("❌ No path found in output object")
                logging.error(f"   Object keys (if dict): {output_object.keys() if isinstance(output_object, dict) else 'N/A'}")
                logging.error(f"   Object attributes: {dir(output_object) if hasattr(output_object, '__dict__') else 'N/A'}")
                continue
                
            logging.info(f"📂 Downloaded file from Sieve: {temp_file_path}")
            logging.info(f"   File exists: {os.path.exists(temp_file_path) if temp_file_path else False}")
            
            # Extract metadata if available
            metadata = {}
            if isinstance(output_object, dict):
                metadata = output_object.get('metadata', {})
                logging.info(f"   Metadata (dict): {metadata}")
            else:
                metadata = getattr(output_object, 'metadata', {})
                logging.info(f"   Metadata (attr): {metadata}")
                
            if isinstance(metadata, dict) and not metadata:
                # If metadata is empty, check if it's directly in the output_object
                if isinstance(output_object, dict) and 'title' in output_object:
                    metadata = output_object
                    logging.info(f"   Using output_object as metadata: {metadata}")
            
            # Get or generate title
            title = metadata.get('title', f"youtube_{os.path.basename(temp_file_path)}")
            logging.info(f"📝 Video title: {title}")
            
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
                "author": metadata.get("channel_id", "Unknown"),  # Use channel_id instead of author
                "length_seconds": metadata.get("duration", 0),    # Use duration instead of length_seconds
                "thumbnail_url": metadata.get("thumbnail", ""),   # Use thumbnail instead of thumbnail_url
                "resolution": "unknown"  # Sieve might not provide this information
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