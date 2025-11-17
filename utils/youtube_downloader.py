import os
import logging
import asyncio
from pathlib import Path
from pytubefix import YouTube

async def download_youtube_video(url: str, output_dir: Path) -> tuple:
    """
    Download a YouTube video using pytubefix
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        tuple: (file_path, title, video_info)
    """
    try:
        logging.info(f"🎬 Starting direct YouTube download (pytubefix)")
        logging.info(f"   URL: {url}")
        logging.info(f"   Output dir: {output_dir}")
        logging.info(f"   Output dir exists: {os.path.exists(output_dir)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"✅ Output directory ready: {output_dir}")
        
        # Run the download in a separate thread to avoid blocking
        logging.info(f"🔄 Running download in executor thread...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _download_video(url, output_dir))
        
        logging.info(f"📊 Executor completed")
        logging.info(f"   Result: {result}")
        
        return result
    
    except Exception as e:
        logging.error(f"❌ Error in download_youtube_video wrapper")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        logging.error(f"   URL: {url}")
        logging.error(f"   Output dir: {output_dir}")
        logging.exception("Full stack trace:")
        return None, None, None

def _download_video(url: str, output_dir: Path) -> tuple:
    """
    Internal function to download YouTube video (runs in executor)
    """
    try:
        logging.info(f"🔧 Initializing pytubefix YouTube object...")
        logging.info(f"   URL: {url}")
        
        # Initialize YouTube object
        yt = YouTube(url)
        
        logging.info(f"✅ YouTube object created")
        logging.info(f"   Title: {yt.title}")
        logging.info(f"   Author: {yt.author}")
        logging.info(f"   Length: {yt.length} seconds")
        logging.info(f"   Views: {yt.views if hasattr(yt, 'views') else 'N/A'}")
        
        # Get the highest resolution progressive stream (includes both video and audio)
        logging.info(f"🔍 Searching for progressive streams...")
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logging.warning("⚠️ No progressive stream found, trying to get highest resolution video")
            # If no progressive stream, get the highest resolution video
            stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logging.error("❌ No suitable video stream found")
            logging.error(f"   Available streams: {len(yt.streams)}")
            for s in yt.streams:
                logging.error(f"     - {s}")
            return None, None, None
        
        logging.info(f"✅ Stream selected: {stream}")
        logging.info(f"   Resolution: {stream.resolution}")
        logging.info(f"   FPS: {stream.fps if hasattr(stream, 'fps') else 'N/A'}")
        logging.info(f"   File size: {stream.filesize:,} bytes ({stream.filesize/1024/1024:.2f} MB)")
        
        # Download the video
        logging.info(f"⬇️ Starting download: {yt.title} ({stream.resolution})")
        logging.info(f"   Output directory: {output_dir}")
        file_path = stream.download(output_path=str(output_dir))
        
        logging.info(f"✅ Download completed: {file_path}")
        logging.info(f"   File exists: {os.path.exists(file_path)}")
        logging.info(f"   File size: {os.path.getsize(file_path):,} bytes")
        
        # Get video info
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "length_seconds": yt.length,
            "thumbnail_url": yt.thumbnail_url,
            "resolution": stream.resolution,
        }
        
        logging.info(f"📊 Video info collected: {video_info}")
        return file_path, yt.title, video_info
        
    except Exception as e:
        logging.error(f"❌ Error in _download_video")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Error message: {str(e)}")
        logging.error(f"   URL: {url}")
        logging.error(f"   Output dir: {output_dir}")
        logging.exception("Full stack trace:")
        return None, None, None 