#!/usr/bin/env python3
"""
Configuration loading utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import dataclasses


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Path to YAML configuration file
    
    Returns:
        Dictionary with configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_yaml_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save YAML file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def config_to_dataclass(config_dict: Dict[str, Any], dataclass_type: Any) -> Any:
    """
    Convert dictionary to dataclass instance.
    
    Args:
        config_dict: Configuration dictionary
        dataclass_type: Dataclass type
    
    Returns:
        Instance of dataclass_type
    """
    if not dataclasses.is_dataclass(dataclass_type):
        raise TypeError(f"Expected dataclass, got {type(dataclass_type)}")
    
    field_names = {f.name for f in dataclasses.fields(dataclass_type)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    
    return dataclass_type(**filtered_dict)


def get_default_config_path() -> Optional[str]:
    """
    Get default configuration file path.
    
    Returns:
        Path to default config file if exists, else None
    """
    possible_paths = [
        "./configs/config.yaml",
        "./configs/advanced_training_config.yaml",
        "./config.yaml"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None
