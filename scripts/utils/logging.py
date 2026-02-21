#!/usr/bin/env python3
"""
Logging utilities for CT enhancement scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to add console handler
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_script_logger(script_name: str, log_dir: str = "./logs") -> logging.Logger:
    """
    Get a logger configured for a specific script.
    
    Args:
        script_name: Name of the script (without extension)
        log_dir: Directory to store log files
    
    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{script_name}.log"
    
    return setup_logger(
        name=script_name,
        log_file=str(log_file),
        level=logging.INFO,
        console=True
    )


# Default logger for immediate use
logger = setup_logger("ct_enhancement")
