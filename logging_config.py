"""
Logging configuration for Clip-It backend
Configures logging to both console and file with rotation
"""
import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging(log_dir="logs", log_file="backend.log", log_level=logging.INFO):
    """
    Set up logging to both console and rotating file
    
    Args:
        log_dir: Directory to store log files
        log_file: Name of the log file
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Full path to log file
    log_file_path = log_path / log_file
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    logging.info(f"Logging initialized. Log file: {log_file_path}")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return str(log_file_path)
