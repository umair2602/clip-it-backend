# Incremental Progress Tracking Implementation

## Overview

The unified progress tracking system now supports **smooth, incremental progress updates** within each pipeline stage, providing users with real-time feedback during long-running operations.

## Progress Flow

The system now flows smoothly through these stages with incremental updates:

```
0% (QUEUED)
  ↓ gradual increase
5-15% (DOWNLOADING) - ~3 minutes
  ↓ instant jump
25-50% (TRANSCRIBING) - ~7 minutes  
  ↓ instant jump
50-70% (ANALYZING) - ~2.5 minutes
  ↓ instant jump
70-90% (PROCESSING) - ~4 minutes
  ↓ instant jump
90-100% (UPLOADING_CLIPS) - ~2 minutes
  ↓ instant jump
100% (COMPLETED)
```

## Key Features

### 1. Stage-Based Progress Mapping

Each pipeline stage has a defined progress percentage range:

| Stage | Start % | End % | Duration Estimate |
|-------|---------|-------|-------------------|
| QUEUED | 0% | 0% | Instant |
| DOWNLOADING | 5% | 15% | ~3 minutes |
| DOWNLOADED | 15% | 15% | Instant |
| TRANSCRIBING | 25% | 50% | ~7 minutes |
| ANALYZING | 50% | 70% | ~2.5 minutes |
| PROCESSING | 70% | 90% | ~4 minutes |
| UPLOADING_CLIPS | 90% | 100% | ~2 minutes |
| COMPLETED | 100% | 100% | Final |

### 2. Incremental Updates Within Stages

The new `update_progress_incremental()` method allows smooth progress updates:

```python
# Update progress to 50% within the DOWNLOADING stage
# This will interpolate between 5% (start) and 15% (end)
# Result: 5% + (15% - 5%) * 0.5 = 10%
progress_tracker.update_progress_incremental(
    job_id, 
    PipelineStage.DOWNLOADING, 
    0.5  # 50% through the stage
)
```

### 3. Automatic Progress Simulation

The `simulate_progress_updates()` helper function automatically updates progress during long operations:

```python
# Simulate progress updates during a 3-minute download
progress_task = asyncio.create_task(
    simulate_progress_updates(
        job_id, 
        PipelineStage.DOWNLOADING, 
        180,  # 3 minutes in seconds
        check_cancelled_func=check_cancel_wrapper
    )
)

try:
    # Perform the actual download
    file_path = await download_youtube_video(url)
finally:
    # Stop the progress simulation
    progress_task.cancel()
```

## Implementation Details

### Progress Tracker Methods

#### `update_progress(job_id, stage)`
Sets progress to the start percentage of a stage (instant jump).

```python
progress_tracker.update_progress(job_id, PipelineStage.TRANSCRIBING)
# Sets progress to 25%
```

#### `update_progress_incremental(job_id, stage, stage_progress)`
Updates progress within a stage (smooth transition).

```python
# 30% through transcribing stage
# 25% + (50% - 25%) * 0.3 = 32.5%
progress_tracker.update_progress_incremental(
    job_id, 
    PipelineStage.TRANSCRIBING, 
    0.3
)
```

#### `update_progress_explicit(job_id, status, progress)`
Sets an explicit progress percentage (for special cases).

```python
progress_tracker.update_progress_explicit(job_id, "processing", 75)
```

### Worker Integration

The worker now uses incremental progress for all long-running operations:

1. **YouTube Download** (5% → 15%, ~3 min)
   - Starts at 5%
   - Gradually increases to 15% during download
   - Jumps to 15% when complete

2. **Transcription** (25% → 50%, ~7 min)
   - Starts at 25%
   - Gradually increases to 50% during transcription
   - Jumps to 50% when complete

3. **AI Analysis** (50% → 70%, ~2.5 min)
   - Starts at 50%
   - Gradually increases to 70% during analysis
   - Jumps to 70% when complete

4. **Video Processing** (70% → 90%, ~4 min)
   - Starts at 70%
   - Gradually increases to 90% during clip creation
   - Jumps to 90% when complete

5. **S3 Upload** (90% → 100%, ~2 min)
   - Starts at 90%
   - Gradually increases to 100% during upload
   - Reaches 100% when complete

## Testing

Run the incremental progress test:

```bash
cd clip-it-backend
python test_incremental_progress.py
```

Expected output:
```
======================================================================
Testing Incremental Progress Tracking
======================================================================

📍 Stage 1: QUEUED (0%)
   Progress: 0% - Status: queued

📍 Stage 2: DOWNLOADING (5% → 15%)
   Start: 5% - Status: downloading
   0% within stage → 5% overall
   20% within stage → 7% overall
   40% within stage → 9% overall
   60% within stage → 11% overall
   80% within stage → 13% overall
   100% within stage → 15% overall

📍 Stage 3: TRANSCRIBING (25% → 50%)
   Start: 25% - Status: transcribing
   0% within stage → 25% overall
   20% within stage → 30% overall
   ...

✅ All incremental progress tests passed!
```

## Benefits

1. **Better User Experience**: Users see continuous progress instead of long periods with no updates
2. **Accurate Estimates**: Progress percentages reflect actual pipeline stages
3. **Cancellation Support**: Progress simulation respects cancellation requests
4. **Flexible**: Easy to adjust duration estimates per stage
5. **Consistent**: All long-running operations use the same progress tracking system

## Configuration

To adjust progress update frequency or duration estimates, modify the values in `worker.py`:

```python
# In simulate_progress_updates()
update_interval = min(2.0, duration_seconds / 10)  # Update every 2 seconds

# In process_youtube_download_job()
simulate_progress_updates(job_id, PipelineStage.DOWNLOADING, 180)  # 3 minutes

# In process_video_job()
simulate_progress_updates(job_id, PipelineStage.TRANSCRIBING, 420)  # 7 minutes
simulate_progress_updates(job_id, PipelineStage.ANALYZING, 150)     # 2.5 minutes
simulate_progress_updates(job_id, PipelineStage.PROCESSING, 240)    # 4 minutes
```

## Migration Notes

- The old progress percentages have been updated:
  - QUEUED: 5% → 0%
  - DOWNLOADING: 8% → 5%
  - TRANSCRIBING: 30% → 25%
  
- All worker functions now use incremental progress
- The `app.py` status endpoint correctly reads progress without updating it
- Frontend polling will now see smooth progress transitions

## Next Steps

1. ✅ Core incremental progress implementation
2. ✅ Worker integration for all stages
3. ✅ Testing and validation
4. 🔄 Monitor real-world performance and adjust duration estimates
5. 🔄 Optional: Add progress callbacks for more granular updates from external services
