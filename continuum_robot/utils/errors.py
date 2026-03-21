"""Shared domain error types."""


class SafetyViolationError(Exception):
    """Raised when a motion command violates configured safety limits."""
