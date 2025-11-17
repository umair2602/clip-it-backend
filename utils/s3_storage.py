import boto3
import os
import logging
import tempfile
import shutil
from botocore.exceptions import ClientError
from pathlib import Path
import uuid
import sys
import asyncio
from typing import Optional, Tuple, Dict, Any
import requests
from urllib.parse import urlparse

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logger = logging.getLogger(__name__)

class S3Client:
    def __init__(self):
        if not all([settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, settings.S3_BUCKET]):
            logger.warning("Missing S3 credentials. Direct upload functionality will not be available.")
            self.available = False
            return
            
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET
        self.available = True
    
    def generate_presigned_post(self, video_id, filename, content_type="video/mp4"):
        """
        Generate a presigned POST URL for S3 upload.
        
        Returns:
            dict: containing url and fields needed for the POST request
        """
        if not self.available:
            return None
            
        try:
            # Define a unique key (path) for the file
            s3_key = f"{settings.S3_UPLOAD_PREFIX}{video_id}/{filename}"
            
            # Generate the presigned post data
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=self.bucket,
                Key=s3_key,
                Fields={
                    "Content-Type": content_type,
                    "x-amz-meta-uploader": "podcast-clipper",
                    "x-amz-meta-video-id": video_id
                },
                Conditions=[
                    ["content-length-range", 1, 5368709120],  # 5GB max
                    ["starts-with", "$Content-Type", ""],
                    ["starts-with", "$x-amz-meta-uploader", ""],
                    ["starts-with", "$x-amz-meta-video-id", ""]
                ],
                ExpiresIn=3600  # 1 hour
            )
            
            if not presigned_post:
                logger.error("Failed to generate presigned post data")
                return None
                
            return {
                "url": presigned_post["url"],
                "fields": presigned_post["fields"],
                "s3_key": s3_key
            }
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None
    
    async def download_youtube_to_s3(self, url: str, video_id: str, title: str = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Download YouTube video directly to S3 without local storage.
        
        Args:
            url: YouTube video URL
            video_id: Unique video identifier
            title: Video title (optional)
            
        Returns:
            Tuple[bool, str, Dict]: (success, s3_key, video_info)
        """
        logger.info(f"="*80)
        logger.info(f"📥 STARTING YOUTUBE DOWNLOAD TO S3")
        logger.info(f"   Video ID: {video_id}")
        logger.info(f"   URL: {url}")
        logger.info(f"   Title: {title}")
        logger.info(f"="*80)
        
        if not self.available:
            logger.error("❌ S3 client not available - missing credentials or configuration")
            logger.error(f"   AWS_ACCESS_KEY_ID set: {bool(settings.AWS_ACCESS_KEY_ID)}")
            logger.error(f"   AWS_SECRET_ACCESS_KEY set: {bool(settings.AWS_SECRET_ACCESS_KEY)}")
            logger.error(f"   S3_BUCKET: {settings.S3_BUCKET}")
            return False, None, {}
        
        logger.info(f"✅ S3 client available")
        logger.info(f"   Bucket: {self.bucket}")
        logger.info(f"   Region: {settings.AWS_REGION}")
            
        try:
            # Import YouTube downloader
            logger.info("📦 Importing YouTube downloader modules...")
            from utils.sieve_downloader import download_youtube_video_sieve
            from utils.youtube_downloader import download_youtube_video
            logger.info("✅ YouTube downloader modules imported successfully")
            
            # Create temporary directory for download
            logger.info(f"📁 Creating temporary directory for download...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"✅ Temporary directory created: {temp_path}")
                
                # Try Sieve first, then fallback to direct download
                file_path, video_title, video_info = None, None, None
                
                try:
                    logger.info(f"🔄 ATTEMPT 1: Downloading with Sieve service")
                    logger.info(f"   URL: {url}")
                    logger.info(f"   Output dir: {temp_path}")
                    file_path, video_title, video_info = await download_youtube_video_sieve(url, temp_path)
                    
                    if file_path and video_title:
                        logger.info(f"✅ Sieve download successful!")
                        logger.info(f"   File: {file_path}")
                        logger.info(f"   Title: {video_title}")
                        logger.info(f"   Info: {video_info}")
                    else:
                        logger.warning(f"⚠️ Sieve returned incomplete data")
                        logger.warning(f"   file_path: {file_path}")
                        logger.warning(f"   video_title: {video_title}")
                        raise Exception("Sieve returned incomplete data")
                        
                except Exception as sieve_error:
                    logger.warning(f"❌ Sieve download failed: {type(sieve_error).__name__}: {str(sieve_error)}")
                    logger.warning(f"   Full error: {repr(sieve_error)}")
                    
                    try:
                        logger.info(f"🔄 ATTEMPT 2: Downloading with direct YouTube downloader (pytubefix)")
                        logger.info(f"   URL: {url}")
                        logger.info(f"   Output dir: {temp_path}")
                        file_path, video_title, video_info = await download_youtube_video(url, temp_path)
                        
                        if file_path and video_title:
                            logger.info(f"✅ Direct download successful!")
                            logger.info(f"   File: {file_path}")
                            logger.info(f"   Title: {video_title}")
                            logger.info(f"   Info: {video_info}")
                        else:
                            logger.error(f"❌ Direct download returned incomplete data")
                            logger.error(f"   file_path: {file_path}")
                            logger.error(f"   video_title: {video_title}")
                            raise Exception("Direct download returned incomplete data")
                            
                    except Exception as direct_error:
                        logger.error(f"❌ Direct download failed: {type(direct_error).__name__}: {str(direct_error)}")
                        logger.error(f"   Full error: {repr(direct_error)}")
                        logger.error(f"💥 BOTH DOWNLOAD METHODS FAILED")
                        return False, None, {}
                
                if not file_path or not video_title:
                    logger.error("❌ No file path or title returned from download")
                    logger.error(f"   file_path: {file_path}")
                    logger.error(f"   video_title: {video_title}")
                    logger.error(f"   video_info: {video_info}")
                    return False, None, {}
                
                # Validate downloaded file
                logger.info(f"🔍 Validating downloaded file...")
                if not os.path.exists(file_path):
                    logger.error(f"❌ File does not exist: {file_path}")
                    return False, None, {}
                
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ File validation passed")
                logger.info(f"   Path: {file_path}")
                logger.info(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                logger.info(f"   Exists: {os.path.exists(file_path)}")
                
                if file_size == 0:
                    logger.error(f"❌ Downloaded file is empty (0 bytes)")
                    return False, None, {}
                
                # Generate S3 key
                logger.info(f"🔑 Generating S3 key...")
                safe_title = "".join(c for c in video_title if c.isalnum() or c in [' ', '-', '_']).rstrip()
                filename = f"{safe_title}.mp4"
                s3_key = f"{settings.S3_UPLOAD_PREFIX}{video_id}/{filename}"
                logger.info(f"✅ S3 key generated: {s3_key}")
                logger.info(f"   Original title: {video_title}")
                logger.info(f"   Safe title: {safe_title}")
                logger.info(f"   Filename: {filename}")
                
                # Upload to S3
                logger.info(f"☁️ UPLOADING TO S3...")
                logger.info(f"   Source: {file_path}")
                logger.info(f"   Bucket: {self.bucket}")
                logger.info(f"   Key: {s3_key}")
                logger.info(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                try:
                    self.s3_client.upload_file(
                        file_path,
                        self.bucket,
                        s3_key,
                        ExtraArgs={
                            'ContentType': 'video/mp4',
                            'Metadata': {
                                'video-id': video_id,
                                'title': video_title,
                                'source': 'youtube',
                                'original-url': url
                            }
                        }
                    )
                    logger.info(f"✅ S3 upload successful!")
                except Exception as upload_error:
                    logger.error(f"❌ S3 upload failed: {type(upload_error).__name__}: {str(upload_error)}")
                    logger.error(f"   Full error: {repr(upload_error)}")
                    logger.error(f"   File path: {file_path}")
                    logger.error(f"   File exists: {os.path.exists(file_path)}")
                    logger.error(f"   File size: {file_size}")
                    raise
                
                # Generate S3 URL
                logger.info(f"🔗 Generating S3 URL...")
                s3_url = self.get_object_url(s3_key)
                logger.info(f"✅ S3 URL: {s3_url}")
                
                # Update video info with S3 details
                video_info.update({
                    's3_key': s3_key,
                    's3_url': s3_url,
                    'filename': filename,
                    'file_size': file_size
                })
                
                logger.info(f"="*80)
                logger.info(f"🎉 YOUTUBE DOWNLOAD TO S3 COMPLETE!")
                logger.info(f"   Video ID: {video_id}")
                logger.info(f"   S3 Key: {s3_key}")
                logger.info(f"   S3 URL: {s3_url}")
                logger.info(f"   File Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                logger.info(f"="*80)
                return True, s3_key, video_info
                
        except Exception as e:
            logger.error(f"="*80)
            logger.error(f"💥 CRITICAL ERROR IN YOUTUBE DOWNLOAD TO S3")
            logger.error(f"   Error Type: {type(e).__name__}")
            logger.error(f"   Error Message: {str(e)}")
            logger.error(f"   Error Repr: {repr(e)}")
            logger.error(f"   Video ID: {video_id}")
            logger.error(f"   URL: {url}")
            logger.error(f"="*80)
            logger.exception("Full stack trace:")
            return False, None, {}
    
    def upload_file_to_s3(self, file_path: str, video_id: str, filename: str, content_type: str = "video/mp4") -> Tuple[bool, str]:
        """
        Upload a local file to S3.
        
        Args:
            file_path: Local file path
            video_id: Unique video identifier
            filename: Name for the file in S3
            content_type: MIME type
            
        Returns:
            Tuple[bool, str]: (success, s3_key)
        """
        if not self.available:
            return False, None
            
        try:
            s3_key = f"{settings.S3_UPLOAD_PREFIX}{video_id}/{filename}"
            
            self.s3_client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'Metadata': {
                        'video-id': video_id,
                        'uploader': 'podcast-clipper'
                    }
                }
            )
            
            logger.info(f"File uploaded to S3: {s3_key}")
            return True, s3_key
            
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            return False, None
    
    def upload_clip_to_s3(self, clip_path: str, video_id: str, clip_id: str, title: str = None) -> Tuple[bool, str]:
        """
        Upload a generated clip to S3.
        
        Args:
            clip_path: Local clip file path
            video_id: Parent video identifier
            clip_id: Unique clip identifier
            title: Clip title (optional)
            
        Returns:
            Tuple[bool, str]: (success, s3_key)
        """
        if not self.available:
            return False, None
            
        try:
            filename = f"clip_{clip_id}.mp4"
            s3_key = f"{settings.S3_OUTPUT_PREFIX}{video_id}/{filename}"
            
            metadata = {
                'video-id': video_id,
                'clip-id': clip_id,
                'type': 'clip'
            }
            
            if title:
                metadata['title'] = title
            
            self.s3_client.upload_file(
                clip_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'Metadata': metadata
                }
            )
            
            logger.info(f"Clip uploaded to S3: {s3_key}")
            return True, s3_key
            
        except Exception as e:
            logger.error(f"Error uploading clip to S3: {str(e)}")
            return False, None
    
    def upload_thumbnail_to_s3(self, thumbnail_path: str, video_id: str, clip_id: str = None) -> Tuple[bool, str]:
        """
        Upload a thumbnail to S3.
        
        Args:
            thumbnail_path: Local thumbnail file path
            video_id: Video identifier
            clip_id: Clip identifier (optional, for clip thumbnails)
            
        Returns:
            Tuple[bool, str]: (success, s3_key)
        """
        if not self.available:
            return False, None
            
        try:
            if clip_id:
                filename = f"clip_{clip_id}_thumbnail.jpg"
                s3_key = f"{settings.S3_OUTPUT_PREFIX}{video_id}/{filename}"
            else:
                filename = f"video_{video_id}_thumbnail.jpg"
                s3_key = f"{settings.S3_OUTPUT_PREFIX}{video_id}/{filename}"
            
            self.s3_client.upload_file(
                thumbnail_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'Metadata': {
                        'video-id': video_id,
                        'type': 'thumbnail'
                    }
                }
            )
            
            logger.info(f"Thumbnail uploaded to S3: {s3_key}")
            return True, s3_key
            
        except Exception as e:
            logger.error(f"Error uploading thumbnail to S3: {str(e)}")
            return False, None
    
    def download_from_s3(self, s3_key, local_path):
        """
        Download a file from S3 to a local path.
        
        Args:
            s3_key: The S3 key (path in bucket)
            local_path: Local path to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available:
            return False
            
        try:
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=local_path
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            return False
            
    def generate_presigned_url(self, s3_key, expires_in=3600):
        """
        Generate a presigned URL for accessing an S3 object.
        
        Args:
            s3_key: The S3 key (path in bucket)
            expires_in: Expiration time in seconds
            
        Returns:
            str: Presigned URL or None if failed
        """
        if not self.available:
            return None
            
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': s3_key
                },
                ExpiresIn=expires_in
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None
            
    def get_object_url(self, s3_key, presigned: bool = False, expires_in: int = 3600):
        """
        Get the URL for an S3 object. If presigned is True, generate a time-limited
        HTTPS URL suitable for external ingestion (e.g., TikTok PULL_FROM_URL).
        """
        if presigned:
            return self.generate_presigned_url(s3_key, expires_in)
        return f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
    
    def is_s3_url(self, url: str) -> bool:
        """
        Check if a URL is an S3 URL.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if S3 URL, False otherwise
        """
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            return (parsed.netloc.endswith('.amazonaws.com') and 
                   's3' in parsed.netloc and 
                   self.bucket in parsed.path)
        except:
            return False
    
    def delete_from_s3(self, s3_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: The S3 key to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available:
            return False
            
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=s3_key
            )
            logger.info(f"File deleted from S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    def list_video_files(self, video_id: str) -> list:
        """
        List all files for a specific video in S3.
        
        Args:
            video_id: Video identifier
            
        Returns:
            list: List of S3 keys
        """
        if not self.available:
            return []
            
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"{settings.S3_UPLOAD_PREFIX}{video_id}/"
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
            
        except Exception as e:
            logger.error(f"Error listing video files: {str(e)}")
            return []

# Initialize the S3 client
s3_client = S3Client() 