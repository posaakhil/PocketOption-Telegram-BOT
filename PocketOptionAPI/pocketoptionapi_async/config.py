"""
Configuration file for the async PocketOption API
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration"""

    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5
    message_timeout: int = 30


@dataclass
class TradingConfig:
    """Trading configuration"""

    min_order_amount: float = 1.0
    max_order_amount: float = 50000.0
    min_duration: int = 60
    max_duration: int = 43200
    max_concurrent_orders: int = 10
    default_timeout: float = 30.0


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    rotation: str = "1 day"
    retention: str = "7 days"
    log_file: str = "pocketoption_async.log"


class Config:
    """Main configuration class"""

    def __init__(self):
        self.connection = ConnectionConfig()
        self.trading = TradingConfig()
        self.logging = LoggingConfig()

        # Load from environment variables
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables"""

        # Connection settings
        self.connection.ping_interval = int(
            os.getenv("PING_INTERVAL", self.connection.ping_interval)
        )
        self.connection.ping_timeout = int(
            os.getenv("PING_TIMEOUT", self.connection.ping_timeout)
        )
        self.connection.max_reconnect_attempts = int(
            os.getenv("MAX_RECONNECT_ATTEMPTS", self.connection.max_reconnect_attempts)
        )

        # Trading settings
        self.trading.min_order_amount = float(
            os.getenv("MIN_ORDER_AMOUNT", self.trading.min_order_amount)
        )
        self.trading.max_order_amount = float(
            os.getenv("MAX_ORDER_AMOUNT", self.trading.max_order_amount)
        )
        self.trading.default_timeout = float(
            os.getenv("DEFAULT_TIMEOUT", self.trading.default_timeout)
        )

        # Logging settings
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.log_file = os.getenv("LOG_FILE", self.logging.log_file)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "connection": {
                "ping_interval": self.connection.ping_interval,
                "ping_timeout": self.connection.ping_timeout,
                "close_timeout": self.connection.close_timeout,
                "max_reconnect_attempts": self.connection.max_reconnect_attempts,
                "reconnect_delay": self.connection.reconnect_delay,
                "message_timeout": self.connection.message_timeout,
            },
            "trading": {
                "min_order_amount": self.trading.min_order_amount,
                "max_order_amount": self.trading.max_order_amount,
                "min_duration": self.trading.min_duration,
                "max_duration": self.trading.max_duration,
                "max_concurrent_orders": self.trading.max_concurrent_orders,
                "default_timeout": self.trading.default_timeout,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "rotation": self.logging.rotation,
                "retention": self.logging.retention,
                "log_file": self.logging.log_file,
            },
        }


# Global configuration instance
config = Config()
