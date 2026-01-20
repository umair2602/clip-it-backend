"""
ZylaLabs YouTube Downloader API Integration

Uses the ZylaLabs API to download YouTube videos.
API Docs: https://zylalabs.com/api-marketplace/video/youtube+downloader+api
"""

import asyncio
import aiohttp
import logging
import re
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from config import settings

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def download_youtube_video_zyla(
    url: str,
    output_dir: Path,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Download a YouTube video using ZylaLabs API.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded video
        
    Returns:
        Tuple of (file_path, title, video_info) or (None, None, None) on failure
    """
    api_key = getattr(settings, 'ZYLA_API_KEY', None) or os.getenv('ZYLA_API_KEY')
    
    if not api_key:
        logger.error("❌ ZYLA_API_KEY not configured")
        raise ValueError("ZYLA_API_KEY not configured in environment")
    
    # Extract video ID from URL
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"❌ Could not extract video ID from URL: {url}")
        raise ValueError(f"Invalid YouTube URL: {url}")
    
    logger.info(f"🚀 [ZylaLabs] Starting YouTube download")
    logger.info(f"   URL: {url}")
    logger.info(f"   Video ID: {video_id}")
    logger.info(f"   Output dir: {output_dir}")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Get download info from ZylaLabs API
        api_url = f"https://zylalabs.com/api/4105/youtube+downloader+api/5876/download+video"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        params = {
            "id": video_id
        }
        
        logger.info(f"📡 Requesting download info from ZylaLabs...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ ZylaLabs API error: {response.status} - {error_text[:200]}")
                    raise Exception(f"ZylaLabs API error: {response.status}")
                
                data = await response.json()
                logger.info(f"📋 ZylaLabs response received")
                logger.debug(f"   Response: {data}")
        
        # Step 2: Extract download URL and title from response
        download_url = None
        title = "YouTube Video"
        
        if isinstance(data, dict):
            # Get title
            title = data.get('title') or data.get('videoTitle') or data.get('name') or title
            logger.info(f"   Title from API: {title}")
            
            # Check status
            if data.get('status') != 'OK':
                logger.error(f"❌ ZylaLabs API returned non-OK status: {data.get('status')}")
                raise Exception(f"ZylaLabs API error: {data.get('status')}")
            
            # Try to find download URL from 'formats' (combined video+audio streams)
            formats = data.get('formats', [])
            logger.info(f"   Found {len(formats)} formats")
            
            # Priority order: 720p mp4, 1080p mp4, any mp4, then any format with URL
            quality_priority = ['720p', '1080p', '480p', '360p']
            
            for target_quality in quality_priority:
                for fmt in formats:
                    mime_type = fmt.get('mimeType', '')
                    quality_label = fmt.get('qualityLabel', '')
                    url = fmt.get('url', '')
                    
                    # Check if it's a video format (not audio-only)
                    if 'video/' in mime_type and url:
                        if target_quality in quality_label:
                            download_url = url
                            logger.info(f"   Found {target_quality} format: {mime_type}")
                            break
                if download_url:
                    break
            
            # Fallback: any video format
            if not download_url:
                for fmt in formats:
                    mime_type = fmt.get('mimeType', '')
                    url = fmt.get('url', '')
                    if 'video/' in mime_type and url:
                        quality_label = fmt.get('qualityLabel', 'unknown')
                        download_url = url
                        logger.info(f"   Using fallback format: {quality_label} - {mime_type}")
                        break
            
            # Direct URL fields as last resort
            if not download_url:
                download_url = data.get('url') or data.get('download_url') or data.get('videoUrl') or data.get('link')
        
        if not download_url:
            logger.error(f"❌ No download URL found in ZylaLabs response")
            logger.error(f"   Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            if 'formats' in data:
                logger.error(f"   First format sample: {data['formats'][0] if data['formats'] else 'empty'}")
            raise Exception("No download URL in ZylaLabs response")
        
        logger.info(f"✅ Got download URL from ZylaLabs")
        logger.info(f"   Title: {title}")
        
        # Step 3: Download the video file
        file_path = await _download_file(download_url, output_dir, title)
        
        if file_path:
            logger.info(f"✅ ZylaLabs download successful: {file_path}")
            video_info = {
                "title": title,
                "video_id": video_id,
                "source": "zyla"
            }
            return file_path, title, video_info
        else:
            logger.error(f"❌ Failed to download video file from ZylaLabs URL")
            return None, None, None
            
    except asyncio.TimeoutError:
        logger.error(f"❌ ZylaLabs API timeout")
        raise Exception("ZylaLabs API timeout")
    except aiohttp.ClientError as e:
        logger.error(f"❌ ZylaLabs connection error: {str(e)}")
        raise Exception(f"ZylaLabs connection error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ ZylaLabs download error: {str(e)}")
        logger.exception("Full stack trace:")
        raise


async def _download_file(url: str, output_dir: Path, title: str) -> Optional[str]:
    """Download video file from URL."""
    try:
        # Sanitize filename
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
