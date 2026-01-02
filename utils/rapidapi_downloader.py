"""
RapidAPI YouTube Downloader Integration

Uses youtube-info-download-api.p.rapidapi.com for reliable YouTube downloads.
This is the PRIMARY download method with the highest priority.
"""

import os
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


async def download_youtube_video_rapidapi(url: str, output_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Download YouTube video using RapidAPI YouTube Info Download API.
    This is the TIER 0 (highest priority) download method.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        tuple: (file_path, title, video_info)
    """
    try:
        logger.info(f"🚀 [TIER 0] Starting RapidAPI YouTube download")
        logger.info(f"   URL: {url}")
        logger.info(f"   Output dir: {output_dir}")
        
        # Get API key from environment
        api_key = os.getenv("RAPIDAPI_YOUTUBE_KEY")
        if not api_key:
            logger.error("❌ RAPIDAPI_YOUTUBE_KEY not found in environment")
            return None, None, None
        
        logger.info(f"   API key configured: {api_key[:10]}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
        
        # Step 1: Get download info from RapidAPI
        logger.info(f"📡 Requesting download info from RapidAPI...")
        download_info = await _get_download_info(url, api_key)
        
        if not download_info:
            logger.error("❌ Failed to get download info from RapidAPI")
            return None, None, None
        
        logger.info(f"✅ Received download info from RapidAPI")
        logger.info(f"   Title: {download_info.get('title', 'Unknown')}")
        logger.info(f"   Duration: {download_info.get('duration', 'Unknown')}")
        
        # Step 2: Download the video file
        video_url = download_info.get('video_url')
        if not video_url:
            logger.error("❌ No video download URL in response")
            return None, None, None
        
        logger.info(f"⬇️  Downloading video file...")
        file_path = await _download_file(video_url, output_dir, download_info.get('title', 'video'))
        
        if not file_path:
            logger.error("❌ Failed to download video file")
            return None, None, None
        
        logger.info(f"✅ RapidAPI download complete: {file_path}")
        
        # Prepare video info
        video_info = {
            'title': download_info.get('title'),
            'duration': download_info.get('duration'),
            'thumbnail': download_info.get('thumbnail'),
            'channel': download_info.get('channel'),
            'source': 'rapidapi'
        }
        
        return file_path, download_info.get('title'), video_info
        
    except Exception as e:
        logger.error(f"❌ Error in RapidAPI YouTube download")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Error message: {str(e)}")
        logger.exception("Full stack trace:")
        return None, None, None


async def _get_download_info(url: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get download information from RapidAPI.
    
    Args:
        url: YouTube video URL
        api_key: RapidAPI key
        
    Returns:
        Dict with download info or None if failed
    """
    try:
        # RapidAPI endpoint
        api_url = "https://youtube-info-download-api.p.rapidapi.com/ajax/download.php"
        
        # Try multiple format options in order of preference (highest quality first)
        format_options = [
            {'format': '1080', 'add_info': '1'},  # Try 1080p quality (HIGHEST)
            {'format': '720', 'add_info': '1'},   # Try 720p quality
            {'format': '480', 'add_info': '1'},   # Try 480p quality
            {'format': '360', 'add_info': '1'},   # Try 360p quality
        ]
        
        headers = {
            'x-rapidapi-host': 'youtube-info-download-api.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        
        # Try each format until one works
        for format_config in format_options:
            try:
                params = {
                    'url': url,
                    'no_merge': 'false',
                    'allow_extended_duration': 'true',
                    **format_config
                }
                
                logger.info(f"📋 RapidAPI request parameters:")
                logger.info(f"   Format: {format_config.get('format')}")
                logger.info(f"   URL: {url}")
                
                # Make async request
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url, params=params, headers=headers, timeout=30) as response:
                        logger.info(f"📡 RapidAPI response status: {response.status}")
                        
                        if response.status == 200:
                            # Parse response
                            data = await response.json()
                            logger.info(f"✅ RapidAPI response received for format {format_config.get('format')}")
                            
                            # This API returns HTML content, not direct download links
                            # We need to use the progress_url to get the actual download link
                            if data.get('success') and 'progress_url' in data:
                                progress_url = data['progress_url']
                                logger.info(f"📡 Polling progress URL to get download link: {progress_url}")
                                
                                # Poll the progress URL to get the actual download link
                                download_url = await _poll_progress_url(progress_url, api_key)
                                
                                if download_url:
                                    logger.info(f"✅ Found download URL with format {format_config.get('format')}: {download_url[:100]}...")
                                    return {
                                        'video_url': download_url,
                                        'title': data.get('title', 'Unknown'),
                                        'duration': data.get('info', {}).get('duration'),
                                        'thumbnail': data.get('info', {}).get('thumbnail'),
                                        'channel': data.get('info', {}).get('channel') or data.get('info', {}).get('uploader')
                                    }
                                else:
                                    logger.warning(f"⚠️ Format {format_config.get('format')}: Failed to get download URL from progress endpoint")
                            else:
                                logger.warning(f"⚠️ Format {format_config.get('format')}: No progress_url in response")
                                logger.warning(f"   Response keys: {list(data.keys())}")
                        else:
                            error_text = await response.text()
                            logger.warning(f"⚠️ Format {format_config.get('format')} returned {response.status}: {error_text[:200]}")
                            # Try next format
                            continue
                            
            except Exception as format_error:
                logger.warning(f"⚠️ Error trying format {format_config.get('format')}: {str(format_error)}")
                continue
        
        # If all formats failed
        logger.error(f"❌ All format options failed for RapidAPI download")
        return None
                    
    except asyncio.TimeoutError:
        logger.error(f"❌ RapidAPI request timed out after 30 seconds")
        return None
    except Exception as e:
        logger.error(f"❌ Error getting download info from RapidAPI: {str(e)}")
        logger.exception("Full stack trace:")
        return None


async def _poll_progress_url(progress_url: str, api_key: str, max_attempts: int = 10) -> Optional[str]:
    """
    Poll the progress URL to get the actual download link.
    
    Args:
        progress_url: The progress URL from initial response
        api_key: RapidAPI key
        max_attempts: Maximum polling attempts
        
    Returns:
        Direct download URL or None
    """
    try:
        headers = {
            'x-rapidapi-host': 'youtube-info-download-api.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_attempts):
                await asyncio.sleep(2)  # Wait 2 seconds between polls
                
                async with session.get(progress_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        progress_data = await response.json()
                        
                        # Check if download is ready
                        if progress_data.get('download_url'):
                            logger.info(f"✅ Download URL ready after {attempt + 1} attempts")
                            return progress_data['download_url']
                        elif progress_data.get('progress') == 100 or progress_data.get('status') == 'finished':
                            # Check different possible fields for the download URL
                            url = (progress_data.get('download_url') or 
                                   progress_data.get('url') or 
                                   progress_data.get('link'))
                            if url:
                                logger.info(f"✅ Download URL ready after {attempt + 1} attempts")
                                return url
                        
                        logger.info(f"⏳ Progress check {attempt + 1}/{max_attempts}: {progress_data.get('progress', 'N/A')}%")
                    else:
                        logger.warning(f"⚠️ Progress URL returned status {response.status}")
            
            logger.error(f"❌ Failed to get download URL after {max_attempts} polling attempts")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error polling progress URL: {str(e)}")
        return None


async def _download_file(url: str, output_dir: Path, title: str) -> Optional[str]:
    """
    Download video file from URL.
    
    Args:
        url: Direct download URL
        output_dir: Directory to save file
        title: Video title for filename
        
    Returns:
        Path to downloaded file or None if failed
    """
    try:
        # Sanitize filename
        import re
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        safe_title = safe_title[:100]  # Limit length
        
        file_path = output_dir / f"{safe_title}.mp4"
        
        logger.info(f"⬇️  Downloading to: {file_path}")
        
        # Download with progress
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status != 200:
                    logger.error(f"❌ Download failed with status {response.status}")
                    return None
                
                # Get file size
                file_size = int(response.headers.get('content-length', 0))
                logger.info(f"   File size: {file_size / (1024*1024):.2f} MB")
                
                # Download in chunks
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) < chunk_size:
                            progress = (downloaded / file_size * 100) if file_size > 0 else 0
                            logger.info(f"   Progress: {downloaded / (1024*1024):.1f} MB ({progress:.1f}%)")
                
                logger.info(f"✅ Download complete: {file_path}")
                return str(file_path)
                
    except Exception as e:
        logger.error(f"❌ Error downloading file: {str(e)}")
        logger.exception("Full stack trace:")
        return None
