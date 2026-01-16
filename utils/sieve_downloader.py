"""
Sieve Downloader Module - TEMPORARILY DISABLED

The sievedata package dependency is not resolving correctly.
This module has been disabled to allow the application to run.
All functions return None/stub values to maintain compatibility.

TODO: Re-enable when sievedata package is fixed
"""
import os
import logging
import asyncio
from pathlib import Path
# import sieve  # DISABLED - dependency not resolving
import shutil
import tempfile
from config import settings
from typing import Optional, Callable

# ============================================================================
# SIEVE DISABLED - The sievedata package is not installing correctly
# ============================================================================
logging.warning("⚠️  Sieve module is DISABLED - sievedata package dependency not resolving")

# Global cancellation flag for the current download (kept for compatibility)
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
    Download a YouTube video using Sieve API - DISABLED
    
    This function is temporarily disabled because the sievedata package
    dependency is not resolving. Returns None to trigger fallback to
    other download methods (pytubefix, yt-dlp).
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        user_id: Optional user ID
        video_id: Optional video ID
        
    Returns:
        tuple: (None, None, None) - always returns None while disabled
    """
    logging.warning("⚠️  Sieve downloader is DISABLED - returning None to trigger fallback")
    return None, None, None


# ============================================================================
# NOTE: Original sieve implementation has been removed due to dependency issues.
# The sievedata package is not installing correctly in the Docker container.
# 
# When the sievedata package is fixed, restore the original implementation from:
# git history or the backup in this commit message.
#
# The original implementation included:
#   - download_youtube_video_sieve(): Main async download function
#   - _download_video_sieve(): Internal sync download helper
#   - Sieve API integration for YouTube video downloading
#   - Metadata extraction (title, duration, thumbnail)
#   - Cancellation support
# ============================================================================