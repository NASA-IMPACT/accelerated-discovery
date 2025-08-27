"""
Logging utilities for the AKD framework.
"""

import sys
from typing import Optional

from loguru import logger

# Conditional import to avoid errors if config doesn't exist
try:
    from akd.configs.data_search_config import get_config

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

    def get_config():
        # Return a simple mock config if the data search config isn't available
        class MockConfig:
            debug = False

        return MockConfig()


class LoggingConfig:
    """Centralized logging configuration."""

    def __init__(self):
        self.config = get_config()
        self._setup_loggers()

    def _setup_loggers(self):
        """Configure loggers with appropriate levels and formats."""
        # Remove default logger
        logger.remove()

        # Determine log level based on environment
        if self.config.debug:
            log_level = "DEBUG"
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        else:
            log_level = "INFO"
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

        # Add console logger
        logger.add(
            sys.stdout,
            level=log_level,
            format=format_string,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Add file logger for errors (always enabled)
        logger.add(
            "logs/akd_errors.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
        )

        # Add debug file logger in development
        if self.config.debug:
            logger.add(
                "logs/akd_debug.log",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="50 MB",
                retention="7 days",
                compression="gz",
            )


# Global logging configuration
_logging_config = None


def setup_logging():
    """Initialize logging configuration."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance with proper configuration.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    setup_logging()
    if name:
        return logger.bind(name=name)
    return logger


def log_search_event(
    search_id: str,
    event: str,
    data: dict = None,
    level: str = "INFO",
):
    """
    Log search-related events separately from progress updates.

    Args:
        search_id: Unique search identifier
        event: Event type/name
        data: Additional event data
        level: Log level
    """
    setup_logging()
    message = f"[SEARCH:{search_id}] {event}"
    if data:
        message += f" | Data: {data}"

    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def log_component_action(
    component: str,
    action: str,
    details: dict = None,
    level: str = "DEBUG",
):
    """
    Log component-specific actions for debugging.

    Args:
        component: Component name (e.g., "CMRDataSearchAgent", "WebSocketHandler")
        action: Action being performed
        details: Additional details
        level: Log level
    """
    setup_logging()
    message = f"[{component}] {action}"
    if details:
        message += f" | {details}"

    log_func = getattr(logger, level.lower(), logger.debug)
    log_func(message)


def log_api_request(
    method: str,
    url: str,
    status_code: int = None,
    duration_ms: float = None,
):
    """
    Log API requests and responses.

    Args:
        method: HTTP method
        url: Request URL
        status_code: Response status code
        duration_ms: Request duration in milliseconds
    """
    setup_logging()
    message = f"[API] {method} {url}"
    if status_code:
        message += f" -> {status_code}"
    if duration_ms:
        message += f" ({duration_ms:.1f}ms)"

    if status_code and status_code >= 400:
        logger.warning(message)
    else:
        logger.info(message)


def log_websocket_event(search_id: str, event: str, connection_status: str = None):
    """
    Log WebSocket events separately from application logs.

    Args:
        search_id: Search identifier
        event: WebSocket event
        connection_status: Current connection status
    """
    setup_logging()
    message = f"[WS:{search_id}] {event}"
    if connection_status:
        message += f" | Status: {connection_status}"

    logger.debug(message)


class ContextualLogger:
    """Logger with automatic context information."""

    def __init__(self, component: str, search_id: str = None):
        self.component = component
        self.search_id = search_id
        self.logger = get_logger(component)

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log("error", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with context."""
        context_parts = [f"[{self.component}]"]
        if self.search_id:
            context_parts.append(f"[{self.search_id}]")

        full_message = " ".join(context_parts) + f" {message}"

        if kwargs:
            full_message += f" | {kwargs}"

        log_func = getattr(self.logger, level)
        log_func(full_message)
