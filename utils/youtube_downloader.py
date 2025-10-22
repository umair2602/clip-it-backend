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
        logging.info(f"Downloading YouTube video: {url}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the download in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _download_video(url, output_dir))
    
    except Exception as e:
        logging.error(f"Error downloading YouTube video: {str(e)}", exc_info=True)
        return None, None, None

def _download_video_to_s3(url: str, s3_key: str, s3_client) -> tuple:
    """
    Internal function to download YouTube video (runs in executor)
    """
    try:
        # Initialize YouTube object
        yt = YouTube(url)
        
        # Get the highest resolution progressive stream (includes both video and audio)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logging.warning("No progressive stream found, trying to get highest resolution video")
            # If no progressive stream, get the highest resolution video
            stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logging.error("No suitable video stream found")
            return None, None, None
        
        # Download the video
        logging.info(f"Downloading video: {yt.title} ({stream.resolution})")
        file_path = stream.download(output_path=str(output_dir))
        
        # Get video info
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "length_seconds": yt.length,
            "thumbnail_url": yt.thumbnail_url,
            "resolution": stream.resolution,
        }
        
        logging.info(f"Download completed: {file_path}")
        return file_path, yt.title, video_info
        
    except Exception as e:
        logging.error(f"Error in _download_video: {str(e)}", exc_info=True)
        return None, None, None 