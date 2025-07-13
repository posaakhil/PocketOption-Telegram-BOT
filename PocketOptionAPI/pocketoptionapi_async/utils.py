"""
Utility functions for the PocketOption API
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from .models import Candle, OrderResult


def format_session_id(
    session_id: str,
    is_demo: bool = True,
    uid: int = 0,
    platform: int = 1,
    is_fast_history: bool = True,
) -> str:
    """
    Format session ID for authentication

    Args:
        session_id: Raw session ID
        is_demo: Whether this is a demo account
        uid: User ID
        platform: Platform identifier (1=web, 3=mobile)
        is_fast_history: Enable fast history loading

    Returns:
        str: Formatted session message
    """
    import json

    auth_data = {
        "session": session_id,
        "isDemo": 1 if is_demo else 0,
        "uid": uid,
        "platform": platform,
    }

    if is_fast_history:
        auth_data["isFastHistory"] = True

    return f'42["auth",{json.dumps(auth_data)}]'


def calculate_payout_percentage(
    entry_price: float, exit_price: float, direction: str, payout_rate: float = 0.8
) -> float:
    """
    Calculate payout percentage for an order

    Args:
        entry_price: Entry price
        exit_price: Exit price
        direction: Order direction ('call' or 'put')
        payout_rate: Payout rate (default 80%)

    Returns:
        float: Payout percentage
    """
    if direction.lower() == "call":
        win = exit_price > entry_price
    else:  # put
        win = exit_price < entry_price

    return payout_rate if win else -1.0


def analyze_candles(candles: List[Candle]) -> Dict[str, Any]:
    """
    Analyze candle data for basic statistics

    Args:
        candles: List of candle data

    Returns:
        Dict[str, Any]: Analysis results
    """
    if not candles:
        return {}

    prices = [candle.close for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]

    return {
        "count": len(candles),
        "first_price": prices[0],
        "last_price": prices[-1],
        "price_change": prices[-1] - prices[0],
        "price_change_percent": ((prices[-1] - prices[0]) / prices[0]) * 100,
        "highest": max(highs),
        "lowest": min(lows),
        "average_close": sum(prices) / len(prices),
        "volatility": calculate_volatility(prices),
        "trend": determine_trend(prices),
    }


def calculate_volatility(prices: List[float], periods: int = 14) -> float:
    """
    Calculate price volatility (standard deviation)

    Args:
        prices: List of prices
        periods: Number of periods for calculation

    Returns:
        float: Volatility value
    """
    if len(prices) < periods:
        periods = len(prices)

    recent_prices = prices[-periods:]
    mean = sum(recent_prices) / len(recent_prices)

    variance = sum((price - mean) ** 2 for price in recent_prices) / len(recent_prices)
    return variance**0.5


def determine_trend(prices: List[float], periods: int = 10) -> str:
    """
    Determine price trend direction

    Args:
        prices: List of prices
        periods: Number of periods to analyze

    Returns:
        str: Trend direction ('bullish', 'bearish', 'sideways')
    """
    if len(prices) < periods:
        periods = len(prices)

    if periods < 2:
        return "sideways"

    recent_prices = prices[-periods:]
    first_half = recent_prices[: periods // 2]
    second_half = recent_prices[periods // 2 :]

    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)

    change_percent = ((second_avg - first_avg) / first_avg) * 100

    if change_percent > 0.1:
        return "bullish"
    elif change_percent < -0.1:
        return "bearish"
    else:
        return "sideways"


def calculate_support_resistance(
    candles: List[Candle], periods: int = 20
) -> Dict[str, float]:
    """
    Calculate support and resistance levels

    Args:
        candles: List of candle data
        periods: Number of periods to analyze

    Returns:
        Dict[str, float]: Support and resistance levels
    """
    if len(candles) < periods:
        periods = len(candles)

    recent_candles = candles[-periods:]
    highs = [candle.high for candle in recent_candles]
    lows = [candle.low for candle in recent_candles]

    # Simple support/resistance calculation
    resistance = max(highs)
    support = min(lows)

    return {"support": support, "resistance": resistance, "range": resistance - support}


def format_timeframe(seconds: int) -> str:
    """
    Format timeframe seconds to human readable string

    Args:
        seconds: Timeframe in seconds

    Returns:
        str: Formatted timeframe (e.g., '1m', '5m', '1h')
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    elif seconds < 86400:
        return f"{seconds // 3600}h"
    else:
        return f"{seconds // 86400}d"


def validate_asset_symbol(symbol: str, available_assets: Dict[str, int]) -> bool:
    """
    Validate if asset symbol is available

    Args:
        symbol: Asset symbol to validate
        available_assets: Dictionary of available assets

    Returns:
        bool: True if asset is available
    """
    return symbol in available_assets


def calculate_order_expiration(
    duration_seconds: int, current_time: Optional[datetime] = None
) -> datetime:
    """
    Calculate order expiration time

    Args:
        duration_seconds: Duration in seconds
        current_time: Current time (default: now)

    Returns:
        datetime: Expiration time
    """
    if current_time is None:
        current_time = datetime.now()

    return current_time + timedelta(seconds=duration_seconds)


def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator for retrying async functions

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff_factor: Delay multiplier for each attempt
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        return wrapper

    return decorator


def performance_monitor(func):
    """
    Decorator to monitor function performance
    """

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


class RateLimiter:
    """
    Rate limiter for API calls
    """

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    async def acquire(self) -> bool:
        """
        Acquire permission to make a call

        Returns:
            bool: True if permission granted
        """
        now = time.time()

        # Remove old calls outside time window
        self.calls = [
            call_time for call_time in self.calls if now - call_time < self.time_window
        ]

        # Check if we can make another call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True

        # Calculate wait time
        wait_time = self.time_window - (now - self.calls[0])
        if wait_time > 0:
            logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            return await self.acquire()

        return True


class OrderManager:
    """
    Manage multiple orders and their results
    """

    def __init__(self):
        self.active_orders: Dict[str, OrderResult] = {}
        self.completed_orders: Dict[str, OrderResult] = {}
        self.order_callbacks: Dict[str, List] = {}

    def add_order(self, order: OrderResult) -> None:
        """Add an active order"""
        self.active_orders[order.order_id] = order

    def complete_order(self, order_id: str, result: OrderResult) -> None:
        """Mark order as completed"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]

        self.completed_orders[order_id] = result

        # Call any registered callbacks
        if order_id in self.order_callbacks:
            for callback in self.order_callbacks[order_id]:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in order callback: {e}")
            del self.order_callbacks[order_id]

    def add_order_callback(self, order_id: str, callback) -> None:
        """Add callback for order completion"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append(callback)

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get order status"""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        return None

    def get_active_count(self) -> int:
        """Get number of active orders"""
        return len(self.active_orders)

    def get_completed_count(self) -> int:
        """Get number of completed orders"""
        return len(self.completed_orders)


def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    """
    Convert candles to pandas DataFrame

    Args:
        candles: List of candle objects

    Returns:
        pd.DataFrame: Candles as DataFrame
    """
    data = []
    for candle in candles:
        data.append(
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "asset": candle.asset,
            }
        )

    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

    return df
