"""
Serialization utilities for consistent JSON handling across the application.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Pydantic models and common Python types safely."""

    def default(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_none=True)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif hasattr(obj, "_asdict"):  # namedtuple
            return obj._asdict()
        else:
            return str(obj)


def safe_serialize(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False, **kwargs)


def safe_serialize_for_websocket(obj: Any) -> Dict[str, Any]:
    """
    Prepare an object for WebSocket transmission by converting to JSON-safe dict.

    Args:
        obj: Object to prepare for WebSocket

    Returns:
        Dictionary ready for JSON serialization
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    elif isinstance(obj, dict):
        return {key: safe_serialize_for_websocket(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize_for_websocket(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return safe_serialize_for_websocket(obj.__dict__)
    elif hasattr(obj, "_asdict"):  # namedtuple
        return safe_serialize_for_websocket(obj._asdict())
    else:
        return obj


def safe_model_dump(obj: Any, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Safely convert objects to dictionaries, handling both Pydantic models and regular objects.

    Args:
        obj: Object to convert to dictionary
        exclude_none: Whether to exclude None values from Pydantic models

    Returns:
        Dictionary representation of the object
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=exclude_none)
    elif isinstance(obj, dict):
        return dict(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif hasattr(obj, "_asdict"):  # namedtuple
        return obj._asdict()
    else:
        # For simple types, wrap in a dict
        return {"value": obj}


def safe_model_dump_list(
    objects: List[Any],
    exclude_none: bool = True,
) -> List[Dict[str, Any]]:
    """
    Safely convert a list of objects to a list of dictionaries.

    Args:
        objects: List of objects to convert
        exclude_none: Whether to exclude None values from Pydantic models

    Returns:
        List of dictionary representations
    """
    return [safe_model_dump(obj, exclude_none=exclude_none) for obj in objects]


def validate_websocket_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean a WebSocket message dictionary.

    Args:
        message: Message dictionary to validate

    Returns:
        Validated and cleaned message

    Raises:
        ValueError: If message format is invalid
    """
    if not isinstance(message, dict):
        raise ValueError("WebSocket message must be a dictionary")

    required_fields = {"type", "search_id", "timestamp"}
    missing_fields = required_fields - set(message.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Ensure all values are JSON serializable
    try:
        safe_serialize(message)
    except Exception as e:
        raise ValueError(f"Message contains non-serializable data: {e}")

    return message


def create_progress_message(
    event_type: str,
    search_id: str,
    data: Dict[str, Any],
    current_step: int = 0,
    total_steps: int = 7,
) -> Dict[str, Any]:
    """
    Create a standardized progress message for WebSocket transmission.

    Args:
        event_type: Type of progress event
        search_id: Unique search identifier
        data: Event-specific data payload
        current_step: Current workflow step
        total_steps: Total number of workflow steps

    Returns:
        Standardized progress message dictionary
    """
    message = {
        "type": event_type,
        "search_id": search_id,
        "timestamp": datetime.now().isoformat(),
        "progress": {
            "current_step": current_step,
            "total_steps": total_steps,
            "percentage": int((current_step / total_steps) * 100)
            if total_steps > 0
            else 0,
        },
        "data": safe_serialize_for_websocket(data),
    }

    return validate_websocket_message(message)


def create_error_message(
    search_id: str,
    error: str,
    failed_step: str = None,
) -> Dict[str, Any]:
    """
    Create a standardized error message for WebSocket transmission.

    Args:
        search_id: Unique search identifier
        error: Error message
        failed_step: Optional step where error occurred

    Returns:
        Standardized error message dictionary
    """
    data = {
        "message": f"Search failed: {error}",
        "error": error,
    }

    if failed_step:
        data["failed_step"] = failed_step

    message = {
        "type": "search_error",
        "search_id": search_id,
        "timestamp": datetime.now().isoformat(),
        "progress": {
            "current_step": -1,
            "total_steps": 7,
            "percentage": 0,
        },
        "data": data,
    }

    return validate_websocket_message(message)
