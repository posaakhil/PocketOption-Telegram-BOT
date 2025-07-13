"""
Custom exceptions for the PocketOption API
"""


class PocketOptionError(Exception):
    """Base exception for all PocketOption API errors"""

    from typing import Optional

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ConnectionError(PocketOptionError):
    """Raised when connection to PocketOption fails"""

    pass


class AuthenticationError(PocketOptionError):
    """Raised when authentication fails"""

    pass


class OrderError(PocketOptionError):
    """Raised when an order operation fails"""

    pass


class TimeoutError(PocketOptionError):
    """Raised when an operation times out"""

    pass


class InvalidParameterError(PocketOptionError):
    """Raised when invalid parameters are provided"""

    pass


class WebSocketError(PocketOptionError):
    """Raised when WebSocket operations fail"""

    pass
