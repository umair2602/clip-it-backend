"""
Unified Progress Tracking System

This module provides centralized progress and status management for the video
processing pipeline. It defines all valid pipeline stages, their associated
progress percentages, and provides consistent APIs for updating and querying
progress across the application.
"""

import logging
import asyncio
import time
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Enumeration of all valid pipeline stages."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    UPLOADING = "uploading"
    PROCESSING_STARTED = "processing_started"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    UPLOADING_CLIPS = "uploading_clips"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    CANCELLED = "cancelled"
    
    # Social media posting stages
    POSTING_TIKTOK = "posting_tiktok"
    POSTING_INSTAGRAM = "posting_instagram"
    POSTING_COMPLETED = "posting_completed"


@dataclass
class ProgressInfo:
    """Container for progress information."""
    status: str
    progress: int
    
    def __post_init__(self):
        """Validate progress values after initialization."""
        if not isinstance(self.progress, int):
            raise ValueError(f"Progress must be an integer, got {type(self.progress).__name__}")
        if not 0 <= self.progress <= 100:
            raise ValueError(f"Progress must be between 0 and 100, got {self.progress}")


class ProgressTracker:
    """Centralized progress tracking system for the video processing pipeline."""
    
    # Stage to progress percentage mapping
    STAGE_PROGRESS: Dict[PipelineStage, int] = {
        PipelineStage.QUEUED: 0,
        PipelineStage.DOWNLOADING: 0,  # Merged with Queued (starts at 0)
        PipelineStage.DOWNLOADED: 35,
        PipelineStage.UPLOADING: 38,
        PipelineStage.PROCESSING_STARTED: 40,
        PipelineStage.TRANSCRIBING: 45,
        PipelineStage.ANALYZING: 65,
        PipelineStage.PROCESSING: 80,
        PipelineStage.UPLOADING_CLIPS: 90,
        PipelineStage.COMPLETED: 100,
        PipelineStage.FAILED: 0,
        PipelineStage.ERROR: 0,
        PipelineStage.CANCELLED: 0,
        PipelineStage.POSTING_TIKTOK: 85,
        PipelineStage.POSTING_INSTAGRAM: 90,
        PipelineStage.POSTING_COMPLETED: 100,
    }
    
    # Expected durations for each stage in seconds (for auto-increment)
    STAGE_DURATIONS: Dict[PipelineStage, int] = {
        PipelineStage.DOWNLOADING: 30,
        PipelineStage.TRANSCRIBING: 20,
        PipelineStage.ANALYZING: 30,
        PipelineStage.PROCESSING: 60,
        PipelineStage.UPLOADING_CLIPS: 20,
        PipelineStage.POSTING_TIKTOK: 14,
        PipelineStage.POSTING_INSTAGRAM: 14,
    }
    
    def __init__(self, job_manager):
        """
        Initialize the Progress Tracker with a job manager.
        
        Args:
            job_manager: Job queue manager for persistence (e.g., JobQueue instance)
        """
        self.job_manager = job_manager
        self._active_timers: Dict[str, asyncio.Task] = {}
        logger.info("ProgressTracker initialized")
    
    def get_progress(self, job_id: str) -> Optional[ProgressInfo]:
        """
        Retrieve current progress for a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            ProgressInfo with status and progress, or None if job not found
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            logger.debug(f"Job {job_id} not found")
            return None
        
        status = job.get('status', 'queued')
        progress = job.get('progress', 0)
        
        # Ensure progress is an integer (handle legacy string values)
        if isinstance(progress, str):
            try:
                progress = int(progress)
            except ValueError:
                logger.warning(f"Invalid progress value '{progress}' for job {job_id}, defaulting to 0")
                progress = 0
        
        return ProgressInfo(status=status, progress=progress)
    
    def update_progress(self, job_id: str, stage: PipelineStage, original_job_id: Optional[str] = None) -> ProgressInfo:
        """
        Update job progress to a specific pipeline stage.
        
        Args:
            job_id: The job identifier
            stage: The pipeline stage to set
            original_job_id: Optional original job ID to also update
            
        Returns:
            Updated ProgressInfo
            
        Raises:
            ValueError: If stage is invalid
        """
        # Stop any existing auto-increment for this job
        self.stop_auto_increment(job_id)
        if original_job_id:
            self.stop_auto_increment(original_job_id)
        
        if stage not in self.STAGE_PROGRESS:
            raise ValueError(f"Invalid pipeline stage: {stage}")
        
        progress = self.STAGE_PROGRESS[stage]
        status = stage.value
        
        self.job_manager.update_job(job_id, {
            'status': status,
            'progress': progress
        })
        
        if original_job_id:
            self.job_manager.update_job(original_job_id, {
                'status': status,
                'progress': progress
            })
        
        # Start auto-increment for stages that support it
        if stage in self.STAGE_DURATIONS and progress < 100:
            self.start_auto_increment(job_id, stage, original_job_id)
            
        logger.debug(f"Updated job {job_id}: {status} ({progress}%)")
        return ProgressInfo(status=status, progress=progress)
    
    def update_progress_by_status(self, job_id: str, status: str) -> ProgressInfo:
        """
        Update job progress using a status string.
        
        Args:
            job_id: The job identifier
            status: The status string (must match a PipelineStage value)
            
        Returns:
            Updated ProgressInfo
            
        Raises:
            ValueError: If status is invalid
        """
        try:
            stage = PipelineStage(status)
        except ValueError:
            valid_statuses = [s.value for s in PipelineStage]
            raise ValueError(
                f"Invalid status string: '{status}'. Must be one of {valid_statuses}"
            )
        
        return self.update_progress(job_id, stage)
    
    def update_progress_explicit(self, job_id: str, status: str, progress: int) -> ProgressInfo:
        """
        Update job progress with explicit percentage.
        
        Args:
            job_id: The job identifier
            status: The status string
            progress: The progress percentage (0-100)
            
        Returns:
            Updated ProgressInfo
            
        Raises:
            ValueError: If progress is invalid
        """
        if not isinstance(progress, int):
            raise ValueError(f"Progress must be an integer, got {type(progress).__name__}")
        if not 0 <= progress <= 100:
            raise ValueError(f"Progress must be between 0 and 100, got {progress}")
        
        self.job_manager.update_job(job_id, {
            'status': status,
            'progress': progress
        })
        
        logger.debug(f"Updated job {job_id}: {status} ({progress}%)")
        return ProgressInfo(status=status, progress=progress)
    
    def update_progress_incremental(
        self, 
        job_id: str, 
        stage: PipelineStage, 
        stage_progress: float,
        message: Optional[str] = None
    ) -> ProgressInfo:
        """
        Update job progress incrementally within a stage.
        
        This method allows smooth progress updates within a stage by interpolating
        between the current stage's start percentage and the next stage's start percentage.
        
        Args:
            job_id: The job identifier
            stage: The current pipeline stage
            stage_progress: Progress within the stage (0.0 to 1.0)
            message: Optional message to update
            
        Returns:
            Updated ProgressInfo
            
        Raises:
            ValueError: If stage is invalid or stage_progress is out of range
            
        Example:
            # During downloading (5% to 10%), report 50% progress within stage
            tracker.update_progress_incremental(job_id, PipelineStage.DOWNLOADING, 0.5)
            # This will set progress to 7.5% (halfway between 5% and 10%)
        """
        if stage not in self.STAGE_PROGRESS:
            raise ValueError(f"Invalid pipeline stage: {stage}")
        
        if not isinstance(stage_progress, (int, float)):
            raise ValueError(f"stage_progress must be a number, got {type(stage_progress).__name__}")
        
        if not 0.0 <= stage_progress <= 1.0:
            raise ValueError(f"stage_progress must be between 0.0 and 1.0, got {stage_progress}")
        
        # Get current stage percentage
        current_stage_pct = self.STAGE_PROGRESS[stage]
        
        # Find next stage percentage for interpolation
        stage_list = list(PipelineStage)
        try:
            current_idx = stage_list.index(stage)
            # Find next non-terminal stage
            next_stage_pct = current_stage_pct + 5  # Default increment
            for i in range(current_idx + 1, len(stage_list)):
                next_stage = stage_list[i]
                if next_stage in self.STAGE_PROGRESS:
                    next_pct = self.STAGE_PROGRESS[next_stage]
                    if next_pct > current_stage_pct:
                        next_stage_pct = next_pct
                        break
        except (ValueError, IndexError):
            next_stage_pct = min(current_stage_pct + 5, 100)
        
        # Interpolate between current and next stage
        progress = int(current_stage_pct + (next_stage_pct - current_stage_pct) * stage_progress)
        progress = max(current_stage_pct, min(progress, next_stage_pct))  # Clamp to range
        
        status = stage.value
        
        update_data = {
            'status': status,
            'progress': progress
        }
        
        if message:
            update_data['message'] = message
        
        self.job_manager.update_job(job_id, update_data)
        
        logger.debug(f"Updated job {job_id}: {status} ({progress}% - {stage_progress*100:.1f}% within stage)")
        return ProgressInfo(status=status, progress=progress)
    
    def start_auto_increment(self, job_id: str, stage: PipelineStage, original_job_id: Optional[str] = None):
        """
        Start a background task to slowly increment progress for a stage.
        """
        self.stop_auto_increment(job_id)
        
        duration = self.STAGE_DURATIONS.get(stage, 60)
        self._active_timers[job_id] = asyncio.create_task(
            self._increment_timer_task(job_id, stage, duration, original_job_id)
        )
        logger.debug(f"Started auto-increment for job {job_id}, stage {stage.value} (target: {duration}s, linked: {original_job_id})")

    def stop_auto_increment(self, job_id: str):
        """
        Stop any active auto-increment task for a job.
        """
        if job_id in self._active_timers:
            task = self._active_timers.pop(job_id)
            task.cancel()
            logger.debug(f"Stopped auto-increment for job {job_id}")

    async def _increment_timer_task(self, job_id: str, stage: PipelineStage, duration: int, original_job_id: Optional[str] = None):
        """
        The background task that performs incremental updates.
        """
        try:
            steps = 20  # Number of steps per stage for smoothness
            delay = duration / steps
            
            for i in range(1, steps):
                await asyncio.sleep(delay)
                # Calculate internal stage progress (0.05, 0.1, ..., 0.95)
                # We stop at 95% of the stage to leave room for the final jump
                stage_progress = i / steps
                try:
                    self.update_progress_incremental(job_id, stage, stage_progress * 0.95)
                    if original_job_id:
                        self.update_progress_incremental(original_job_id, stage, stage_progress * 0.95)
                except Exception as e:
                    logger.error(f"Error in auto-increment for job {job_id}: {e}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Unexpected error in timer task for job {job_id}: {e}")
    
    def get_stage_progress(self, stage: PipelineStage) -> int:
        """
        Get the progress percentage for a specific stage.
        
        Args:
            stage: The pipeline stage
            
        Returns:
            Progress percentage (0-100)
        """
        return self.STAGE_PROGRESS.get(stage, 0)
    
    @classmethod
    def get_all_stages(cls) -> Dict[str, int]:
        """
        Get all defined stages and their progress percentages.
        
        Returns:
            Dictionary mapping status strings to progress percentages
        """
        return {stage.value: progress for stage, progress in cls.STAGE_PROGRESS.items()}
