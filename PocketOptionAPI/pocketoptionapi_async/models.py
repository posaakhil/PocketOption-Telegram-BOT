"""
Pydantic models for type safety and validation
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class OrderDirection(str, Enum):
    """
    Represents the direction of an order in trading.
    - CALL: A call option, predicting the price will go up.
    - PUT: A put option, predicting the price will go down.
    """

    CALL = "call"
    PUT = "put"


class OrderStatus(str, Enum):
    """
    Represents the current status of a trading order.
    - PENDING: The order has been submitted but not yet processed.
    - ACTIVE: The order is currently active in the market.
    - CLOSED: The order has been closed (either naturally expired or manually).
    - CANCELLED: The order was cancelled before execution or expiry.
    - WIN: The order resulted in a win (profit).
    - LOSE: The order resulted in a loss.
    """

    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    WIN = "win"
    LOSE = "lose"


class ConnectionStatus(str, Enum):
    """
    Represents the connection status to the trading platform.
    - CONNECTED: Successfully connected to the platform.
    - DISCONNECTED: Connection has been lost.
    - CONNECTING: Attempting to establish a connection.
    - RECONNECTING: Attempting to re-establish a lost connection.
    """

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    RECONNECTING = "reconnecting"


class TimeFrame(int, Enum):
    """
    Represents standard timeframes for candlestick data in seconds.
    These values are commonly used in financial charting to aggregate price data
    over specific intervals.
    """

    S1 = 1  # 1 second
    S5 = 5  # 5 seconds
    S10 = 10  # 10 seconds
    S15 = 15  # 15 seconds
    S30 = 30  # 30 seconds
    M1 = 60  # 1 minute
    M5 = 300  # 5 minutes
    M15 = 900  # 15 minutes
    M30 = 1800  # 30 minutes
    H1 = 3600  # 1 hour
    H4 = 14400  # 4 hours
    D1 = 86400  # 1 day
    W1 = 604800  # 1 week
    MN1 = 2592000  # 1 month (approximate, based on 30 days)


class Asset(BaseModel):
    """
    Asset information model.
    Defines the properties of a tradable asset, such as currency pairs or commodities.
    """

    id: str
    name: str
    symbol: str
    is_active: bool = True
    payout: Optional[float] = None

    class Config:
        frozen = True


class Balance(BaseModel):
    """
    Account balance model.
    Provides details about the user's current account balance, currency,
    and whether it's a demo or real account.
    """

    balance: float
    currency: str = "USD"
    is_demo: bool = True
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = True


class Candle(BaseModel):
    """
    OHLC (Open, High, Low, Close) candle data model.
    Represents a single candlestick, which summarizes price movements over a specific timeframe.
    Includes validation to ensure logical consistency of high and low prices.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    asset: str
    timeframe: int  # in seconds, representing the duration of the candle

    @validator("high")
    def high_must_be_valid(cls, v, values):
        """
        Validator to ensure that the 'high' price is never less than the 'low' price.
        This maintains the logical integrity of candlestick data.
        """
        if "low" in values and v < values["low"]:
            raise ValueError("High must be greater than or equal to low")
        return v

    @validator("low")
    def low_must_be_valid(cls, v, values):
        """
        Validator to ensure that the 'low' price is never greater than the 'high' price.
        This maintains the logical integrity of candlestick data.
        """
        if "high" in values and v > values["high"]:
            raise ValueError("Low must be less than or equal to high")
        return v

    class Config:
        frozen = True


class Order(BaseModel):
    """
    Order request model.
    Defines the parameters for placing a new trading order.
    Includes validation for positive amount and minimum duration.
    """

    asset: str
    amount: float
    direction: OrderDirection
    duration: int  # in seconds, how long the order is active
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator("amount")
    def amount_must_be_positive(cls, v):
        """
        Validator to ensure the trading amount is a positive value.
        An amount of zero or less is not valid for an order.
        """
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v

    @validator("duration")
    def duration_must_be_valid(cls, v):
        """
        Validator to ensure the order duration meets a minimum requirement.
        This prevents orders with impractically short durations.
        """
        if v < 5:  # minimum 5 seconds
            raise ValueError("Duration must be at least 5 seconds")
        return v


class OrderResult(BaseModel):
    """
    Order execution result model.
    Provides details about a executed or closed trading order, including its outcome.
    """

    order_id: str
    asset: str
    amount: float
    direction: OrderDirection
    duration: int
    status: OrderStatus
    placed_at: datetime
    expires_at: datetime
    profit: Optional[float] = None
    payout: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        frozen = True


class ServerTime(BaseModel):
    """
    Server time synchronization model.
    Used to synchronize local client time with the trading server's time,
    important for accurate timestamping of trades and events.
    """

    server_timestamp: float
    local_timestamp: float
    offset: float
    last_sync: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = True


class ConnectionInfo(BaseModel):
    """
    Connection information model.
    Provides details about the current connection to the trading platform,
    including URL, region, status, and connection metrics.
    """

    url: str
    region: str
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    last_ping: Optional[datetime] = None
    reconnect_attempts: int = 0

    class Config:
        frozen = True
