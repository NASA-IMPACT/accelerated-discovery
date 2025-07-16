class AKDError(Exception):
    """Base exception for agent-related errors."""

    pass


class SchemaValidationError(AKDError):
    """Raised when schema validation fails."""

    pass
