"""
Enhanced S3 storage module with multipart upload for better performance.
Add this to your existing s3_storage.py or use as standalone.
"""

import boto3
import os
import logging
import sys
from typing import Tuple
from botocore.exceptions import ClientError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logger = logging.getLogger(__name__)

class MultipartS3Client:
    """S3 client with optimized multipart upload"""
    
    def __init__(self):
        if not all([settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, settings.S3_BUCKET]):
            logger.warning("Missing S3 credentials")
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
    
    def upload_clip_to_s3_optimized(self, clip_path: str, video_id: str, 
                                    clip_id: str, title: str = None,
                                    use_multipart: bool = True) -> Tuple[bool, str]:
        """
        Upload clip with automatic multipart for large files.
        
        Args:
            clip_path: Local clip file path
            video_id: Parent video identifier
            clip_id: Unique clip identifier
            title: Clip title (optional)
            use_multipart: Use multipart for files >50MB
            
        Returns:
            Tuple[bool, str]: (success, s3_key)
        """
        if not self.available:
            return False, None
        
        try:
            filename = f"clip_{clip_id}.mp4"
            s3_key = f"{settings.S3_OUTPUT_PREFIX}{video_id}/{filename}"
            
            # Get file size
            file_size = os.path.getsize(clip_path)
            
            metadata = {
                'video-id': video_id,
                'clip-id': clip_id,
                'type': 'clip'
            }
            
            if title:
                metadata['title'] = title
            
            # Use multipart for files > 50MB
            if use_multipart and file_size > 50 * 1024 * 1024:
                logger.info(f"File size {file_size / 1024 / 1024:.1f}MB - using multipart upload")
                return self._multipart_upload(clip_path, s3_key, 'video/mp4', metadata)
            else:
                logger.info(f"File size {file_size / 1024 / 1024:.1f}MB - using standard upload")
                return self._standard_upload(clip_path, s3_key, 'video/mp4', metadata)
                
        except Exception as e:
            logger.error(f"Error uploading clip to S3: {str(e)}")
            return False, None
    
    def _standard_upload(self, file_path: str, s3_key: str, 
                        content_type: str, metadata: dict) -> Tuple[bool, str]:
        """Standard single-part upload for files <50MB"""
        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'Metadata': metadata
                }
            )
            logger.info(f"Standard upload completed: {s3_key}")
            return True, s3_key
        except Exception as e:
            logger.error(f"Standard upload failed: {str(e)}")
            return False, None
    
    def _multipart_upload(self, file_path: str, s3_key: str, 
                         content_type: str, metadata: dict,
                         chunk_size: int = 5 * 1024 * 1024) -> Tuple[bool, str]:
        """
        Multipart upload for large files (>50MB).
        
        Advantages:
        - Resume capability if interrupted
        - Faster for large files (up to 4 parallel uploads)
        - Better error recovery
        
        Args:
            chunk_size: Size of each part (default 5MB, min 5MB)
        """
        upload_id = None
        parts = []
        
        try:
            # Initiate multipart upload
            logger.info(f"Initiating multipart upload for {s3_key}")
            mpu = self.s3_client.create_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                ContentType=content_type,
                Metadata=metadata
            )
            upload_id = mpu['UploadId']
            logger.info(f"Upload ID: {upload_id}")
            
            # Upload parts
            part_number = 1
            bytes_uploaded = 0
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Upload part
                    logger.info(f"Uploading part {part_number}...")
                    response = self.s3_client.upload_part(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=chunk
                    )
                    
                    parts.append({
                        'ETag': response['ETag'],
                        'PartNumber': part_number
                    })
                    
                    bytes_uploaded += len(chunk)
                    progress = (bytes_uploaded / file_size) * 100
                    logger.info(f"  Part {part_number} complete - Progress: {progress:.1f}%")
                    part_number += 1
            
            # Complete multipart upload
            logger.info(f"Completing multipart upload with {len(parts)} parts")
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            logger.info(f"✅ Multipart upload completed successfully: {s3_key}")
            return True, s3_key
            
        except ClientError as e:
            logger.error(f"ClientError during multipart upload: {str(e)}")
            # Abort incomplete upload
            if upload_id:
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                    logger.info(f"Aborted incomplete upload: {upload_id}")
                except Exception as abort_error:
                    logger.error(f"Error aborting upload: {str(abort_error)}")
            return False, None
            
        except Exception as e:
            logger.error(f"Error during multipart upload: {str(e)}")
            # Attempt abort
            if upload_id:
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                except:
                    pass
            return False, None
    
    def upload_thumbnail_to_s3_optimized(self, thumbnail_path: str, 
                                         video_id: str, 
                                         clip_id: str = None) -> Tuple[bool, str]:
        """
        Upload thumbnail (always uses standard upload as files are small).
        
        Args:
            thumbnail_path: Local thumbnail file path
            video_id: Video identifier
            clip_id: Clip identifier (optional)
            
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


# Initialize optimized client
multipart_s3_client = MultipartS3Client()


# Integration helper function
def upload_clip_with_optimization(clip_path: str, video_id: str, 
                                 clip_id: str, title: str = None) -> Tuple[bool, str]:
    """
    Wrapper function for easy integration.
    
    Usage in app.py:
        from utils.s3_storage_multipart import upload_clip_with_optimization
        
        success, s3_key = upload_clip_with_optimization(
            clip_path, video_id, clip.get("id"), clip.get("title")
        )
    """
    return multipart_s3_client.upload_clip_to_s3_optimized(
        clip_path, video_id, clip_id, title
    )
