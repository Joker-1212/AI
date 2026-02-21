#!/usr/bin/env python3
"""
Error handling utilities.
"""

import sys
import traceback
from typing import Optional, Callable, Any
from functools import wraps

from .logging import logger


class ScriptError(Exception):
    """Base exception for script errors."""
    pass


class ConfigError(ScriptError):
    """Configuration related errors."""
    pass


class DataError(ScriptError):
    """Data related errors."""
    pass


class ModelError(ScriptError):
    """Model related errors."""
    pass


def handle_exceptions(
    log_error: bool = True,
    raise_error: bool = False,
    exit_on_error: bool = True,
    exit_code: int = 1
) -> Callable:
    """
    Decorator to handle exceptions in functions.
    
    Args:
        log_error: Whether to log the error
        raise_error: Whether to re-raise the error after handling
        exit_on_error: Whether to exit the program on error
        exit_code: Exit code if exiting
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                
                if raise_error:
                    raise
                
                if exit_on_error:
                    sys.exit(exit_code)
                
                return None
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function, returning default value on error.
    
    Args:
        func: Function to execute
        default: Default value to return on error
        log_error: Whether to log errors
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"Error in {func.__name__ if hasattr(func, '__name__') else 'anonymous'}: {str(e)}")
        return default


def validate_file_exists(filepath: str, error_message: Optional[str] = None) -> None:
    """
    Validate that a file exists.
    
    Args:
        filepath: Path to file
        error_message: Custom error message
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath or not os.path.exists(filepath):
        msg = error_message or f"File not found: {filepath}"
        raise FileNotFoundError(msg)


def validate_directory_exists(dirpath: str, create: bool = False) -> None:
    """
    Validate that a directory exists.
    
    Args:
        dirpath: Path to directory
        create: Whether to create directory if it doesn't exist
    
    Raises:
        FileNotFoundError: If directory doesn't exist and create=False
    """
    if not dirpath:
        raise ValueError("Directory path cannot be empty")
    
    if not os.path.exists(dirpath):
        if create:
            os.makedirs(dirpath, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {dirpath}")


# Import os for file operations
import os
