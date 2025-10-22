"""
S3 URL validation and runtime assertions for ensuring all API responses contain S3 URLs only.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from utils.s3_storage import s3_client

logger = logging.getLogger(__name__)

def is_s3_url(url: str) -> bool:
    """
    Check if a URL is a valid S3 URL.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if valid S3 URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        parsed = urlparse(url)
        # S3 URL format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        
        # Check if S3 client is available
        if not s3_client.available:
            logger.warning("S3 client not available during URL validation")
            # Fallback validation without bucket check
            is_valid = (parsed.scheme in ['http', 'https'] and
                       parsed.netloc.endswith('.amazonaws.com') and
                       's3' in parsed.netloc)
        else:
            is_valid = (parsed.scheme in ['http', 'https'] and
                       parsed.netloc.endswith('.amazonaws.com') and
                       's3' in parsed.netloc and
                       s3_client.bucket in parsed.netloc)
        
        # Debug logging
        if not is_valid:
            logger.debug(f"S3 URL validation failed for: {url}")
            logger.debug(f"  scheme: {parsed.scheme}")
            logger.debug(f"  netloc: {parsed.netloc}")
            logger.debug(f"  bucket: {s3_client.bucket if s3_client.available else 'N/A'}")
            logger.debug(f"  s3_client_available: {s3_client.available}")
            logger.debug(f"  conditions: scheme={parsed.scheme in ['http', 'https']}, "
                        f"amazonaws={parsed.netloc.endswith('.amazonaws.com')}, "
                        f"s3_in_netloc={'s3' in parsed.netloc}, "
                        f"bucket_in_netloc={s3_client.bucket in parsed.netloc if s3_client.available else 'N/A'}")
        
        return is_valid
    except Exception as e:
        logger.error(f"Error parsing URL {url}: {e}")
        return False

def validate_s3_url(url: str, context: str = "") -> bool:
    """
    Validate S3 URL and log error if invalid.
    
    Args:
        url: URL to validate
        context: Context for error logging
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not is_s3_url(url):
        logger.error(f"Invalid S3 URL in {context}: {url}")
        return False
    return True

def assert_s3_url(url: str, context: str = ""):
    """
    Runtime assertion to ensure URL is S3 URL.
    
    Args:
        url: URL to validate
        context: Context for error logging
        
    Raises:
        AssertionError: If URL is not a valid S3 URL
    """
    if not validate_s3_url(url, context):
        raise AssertionError(f"Non-S3 URL detected in {context}: {url}")

def validate_response_urls(response_data: Any, context: str = "") -> bool:
    """
    Recursively validate all URLs in a response object.
    
    Args:
        response_data: Response data to validate
        context: Context for error logging
        
    Returns:
        bool: True if all URLs are valid S3 URLs, False otherwise
    """
    if isinstance(response_data, dict):
        for key, value in response_data.items():
            if isinstance(value, str) and any(url_indicator in key.lower() for url_indicator in ['url', 'link', 'src']):
                if not validate_s3_url(value, f"{context}.{key}"):
                    return False
            elif isinstance(value, (dict, list)):
                if not validate_response_urls(value, f"{context}.{key}"):
                    return False
    elif isinstance(response_data, list):
        for i, item in enumerate(response_data):
            if not validate_response_urls(item, f"{context}[{i}]"):
                return False
    
    return True

def assert_response_urls(response_data: Any, context: str = ""):
    """
    Runtime assertion to ensure all URLs in response are S3 URLs.
    
    Args:
        response_data: Response data to validate
        context: Context for error logging
        
    Raises:
        AssertionError: If any non-S3 URL is found
    """
    if not validate_response_urls(response_data, context):
        raise AssertionError(f"Non-S3 URLs detected in response: {context}")

def sanitize_urls_for_response(response_data: Any) -> Any:
    """
    Sanitize response data to ensure all URLs are S3 URLs.
    This function can be used to clean up any non-S3 URLs before sending response.
    
    Args:
        response_data: Response data to sanitize
        
    Returns:
        Any: Sanitized response data
    """
    if isinstance(response_data, dict):
        sanitized = {}
        for key, value in response_data.items():
            if isinstance(value, str) and any(url_indicator in key.lower() for url_indicator in ['url', 'link', 'src']):
                if not is_s3_url(value):
                    logger.warning(f"Non-S3 URL found in response key '{key}': {value}")
                    # Replace with a placeholder or remove
                    sanitized[key] = None
                else:
                    sanitized[key] = value
            elif isinstance(value, (dict, list)):
                sanitized[key] = sanitize_urls_for_response(value)
            else:
                sanitized[key] = value
        return sanitized
    elif isinstance(response_data, list):
        return [sanitize_urls_for_response(item) for item in response_data]
    else:
        return response_data

def validate_video_data(video_data: Dict[str, Any]) -> bool:
    """
    Validate video data structure for S3 URLs.
    
    Args:
        video_data: Video data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_s3_fields = ['s3_url', 's3_key']
    
    for field in required_s3_fields:
        if field in video_data:
            if not validate_s3_url(video_data[field], f"video_data.{field}"):
                return False
    
    return True

def validate_clip_data(clip_data: Dict[str, Any]) -> bool:
    """
    Validate clip data structure for S3 URLs.
    
    Args:
        clip_data: Clip data dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_s3_fields = ['s3_url', 's3_key']
    
    for field in required_s3_fields:
        if field in clip_data:
            if not validate_s3_url(clip_data[field], f"clip_data.{field}"):
                return False
    
    return True

def ensure_s3_url(url: str, fallback_url: str = None) -> str:
    """
    Ensure a URL is an S3 URL, return fallback if not.
    
    Args:
        url: URL to check
        fallback_url: Fallback S3 URL if original is not valid
        
    Returns:
        str: Valid S3 URL or fallback
    """
    if is_s3_url(url):
        return url
    
    if fallback_url and is_s3_url(fallback_url):
        logger.warning(f"Using fallback S3 URL: {fallback_url}")
        return fallback_url
    
    logger.error(f"No valid S3 URL available: {url}")
    return None 