"""Logging configuration for StockViz application."""

import logging
import os
from typing import Optional

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO in production, DEBUG in dev)
        
    Returns:
        Configured logger instance
    """
    log_level = level or (logging.DEBUG if os.getenv('DEBUG') else logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Only add handlers if they don't exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
