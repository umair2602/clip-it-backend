# Backend Logging System

## Overview
The Clip-It backend now includes a comprehensive logging system that writes all application logs to both the console and rotating log files.

## Log File Location
All logs are written to: `logs/backend.log`

## Features
- **Dual Output**: Logs are written to both console (for development) and file (for debugging)
- **Automatic Rotation**: Log files automatically rotate when they reach 10MB
- **Backup Files**: Keeps the last 5 rotated log files (backend.log.1, backend.log.2, etc.)
- **Timestamped**: All log entries include timestamps in format: `YYYY-MM-DD HH:MM:SS`
- **Log Levels**: INFO, WARNING, ERROR, DEBUG

## Viewing Logs

### Using PowerShell Script (Recommended)
```powershell
# View last 50 lines (default)
.\view-logs.ps1

# View last 100 lines
.\view-logs.ps1 -Lines 100

# Follow logs in real-time (like tail -f)
.\view-logs.ps1 -Follow

# Follow with custom line count
.\view-logs.ps1 -Follow -Lines 100
```

### Using PowerShell Commands
```powershell
# View last 50 lines
Get-Content logs\backend.log -Tail 50

# Follow logs in real-time
Get-Content logs\backend.log -Wait -Tail 50

# View entire log file
Get-Content logs\backend.log

# Search for specific text
Get-Content logs\backend.log | Select-String "error"

# Search for video processing
Get-Content logs\backend.log | Select-String "clip"
```

### Using Command Prompt
```cmd
# View last 50 lines (approximate)
type logs\backend.log | more

# View entire file
type logs\backend.log
```

## What Gets Logged

### Video Upload Process
- File upload to S3
- S3 URL generation
- Database record creation
- Thumbnail generation
- Task creation

### Video Processing
- Download from S3
- Transcription start/completion
- Content analysis (segment detection)
- Clip generation (for each clip)
- S3 upload (for each clip)
- Thumbnail generation (for each clip)
- Database updates

### YouTube Downloads
- Download initiation
- Sieve API calls
- S3 upload
- Processing start

### Errors and Warnings
- Failed S3 uploads
- Transcription errors
- Database connection issues
- API errors
- File processing errors

## Example Log Entries

```
2025-11-04 14:30:25 - root - INFO - Video uploaded successfully to S3: https://s3.amazonaws.com/...
2025-11-04 14:30:30 - root - INFO - Starting podcast processing for task abc-123, video xyz-456
2025-11-04 14:30:35 - root - INFO - Starting transcription for /tmp/video.mp4
2025-11-04 14:31:20 - root - INFO - Transcription completed with 150 segments
2025-11-04 14:31:25 - root - INFO - Starting content analysis to find interesting segments
2025-11-04 14:31:30 - root - INFO - Content analysis completed, found 3 interesting segments
2025-11-04 14:31:35 - root - INFO - Processing clip 1/3: clip-abc-001
2025-11-04 14:31:40 - root - INFO - ✅ Clip uploaded to S3: https://s3.amazonaws.com/...
2025-11-04 14:32:00 - root - INFO - ✅ Successfully saved 3/3 clips to database
```

## Debugging with Logs

### Find Why Only One Clip is Generated
```powershell
# Search for "found" to see how many segments were detected
Get-Content logs\backend.log | Select-String "found.*segments"

# Search for "Processing clip" to see which clips were created
Get-Content logs\backend.log | Select-String "Processing clip"

# Search for clip upload confirmations
Get-Content logs\backend.log | Select-String "Clip uploaded to S3"

# Search for database save confirmations
Get-Content logs\backend.log | Select-String "saved.*clips to database"
```

### Find Errors
```powershell
# All errors
Get-Content logs\backend.log | Select-String "ERROR"

# All warnings
Get-Content logs\backend.log | Select-String "WARNING"

# Failed operations
Get-Content logs\backend.log | Select-String "Failed|failed"
```

### Track a Specific Video
```powershell
# Replace VIDEO_ID with your actual video ID
Get-Content logs\backend.log | Select-String "VIDEO_ID"

# Track a specific task
Get-Content logs\backend.log | Select-String "TASK_ID"
```

## Log Rotation
Log files are automatically rotated when they reach 10MB. The system keeps:
- `backend.log` (current log)
- `backend.log.1` (most recent backup)
- `backend.log.2`
- `backend.log.3`
- `backend.log.4`
- `backend.log.5` (oldest backup)

Older backups are automatically deleted.

## Configuration
The logging configuration is in `logging_config.py`. You can modify:
- `log_dir`: Directory for log files (default: "logs")
- `log_file`: Log file name (default: "backend.log")
- `log_level`: Logging level (default: `logging.INFO`)
- `maxBytes`: Max size before rotation (default: 10MB)
- `backupCount`: Number of backup files to keep (default: 5)

## Troubleshooting

### Log file not created
1. Make sure the backend server is running
2. Check if the `logs/` directory exists
3. Check file permissions

### Can't view logs
1. Navigate to the backend directory: `cd clip-it-backend`
2. Check if the log file exists: `Test-Path logs\backend.log`
3. Try viewing with: `Get-Content logs\backend.log`

### Logs are empty
1. The backend may not have started successfully
2. Check console output for startup errors
3. Verify all dependencies are installed

## Tips
- Use `-Follow` to watch logs in real-time while testing
- Use `Select-String` to filter logs for specific patterns
- Logs persist across server restarts, so you can review historical data
- Check log rotation backups if you need older logs
