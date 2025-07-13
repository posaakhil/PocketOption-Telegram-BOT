"""
Professional test suite for the Async PocketOption API
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from pocketoptionapi_async import (
    AsyncPocketOptionClient,
    OrderDirection,
    OrderStatus,
    Balance,
    Order,
    OrderResult,
    ConnectionError,
    InvalidParameterError,
    ConnectionStatus,
)


class TestAsyncPocketOptionClient:
    """Test suite for AsyncPocketOptionClient"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return AsyncPocketOptionClient(ssid="test_session", is_demo=True, uid=12345)

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket client"""
        mock = AsyncMock()
        mock.is_connected = True
        mock.connection_info = MagicMock()
        mock.connection_info.status = "connected"
        return mock

    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.session_id == "test_session"
        assert client.is_demo is True
        assert client.uid == 12345
        assert client._balance is None

    @pytest.mark.asyncio
    async def test_connect_success(self, client, mock_websocket):
        """Test successful connection"""
        with patch.object(client, "_websocket", mock_websocket):
            mock_websocket.connect.return_value = True

            result = await client.connect()

            assert result is True
            mock_websocket.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, client, mock_websocket):
        """Test connection failure"""
        with patch.object(client, "_websocket", mock_websocket):
            mock_websocket.connect.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_websocket):
        """Test disconnection"""
        with patch.object(client, "_websocket", mock_websocket):
            await client.disconnect()
            mock_websocket.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_balance_success(self, client):
        """Test getting balance"""
        # Set up test balance
        test_balance = Balance(balance=1000.0, currency="USD", is_demo=True)
        client._balance = test_balance

        # Mock websocket as connected
        client._websocket.websocket = MagicMock()
        client._websocket.websocket.closed = False
        client._websocket.connection_info = MagicMock()
        client._websocket.connection_info.status = ConnectionStatus.CONNECTED

        balance = await client.get_balance()

        assert balance.balance == 1000.0
        assert balance.currency == "USD"
        assert balance.is_demo is True

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self, client):
        """Test getting balance when not connected"""
        # Mock websocket as not connected
        client._websocket.websocket = None

        with pytest.raises(ConnectionError):
            await client.get_balance()

    def test_validate_order_parameters_valid(self, client):
        """Test order parameter validation with valid parameters"""
        # Should not raise any exception
        client._validate_order_parameters(
            asset="EURUSD_otc", amount=10.0, direction=OrderDirection.CALL, duration=120
        )

    def test_validate_order_parameters_invalid_asset(self, client):
        """Test order parameter validation with invalid asset"""
        with pytest.raises(InvalidParameterError):
            client._validate_order_parameters(
                asset="INVALID_ASSET",
                amount=10.0,
                direction=OrderDirection.CALL,
                duration=120,
            )

    def test_validate_order_parameters_invalid_amount(self, client):
        """Test order parameter validation with invalid amount"""
        with pytest.raises(InvalidParameterError):
            client._validate_order_parameters(
                asset="EURUSD_otc",
                amount=0.5,  # Too low
                direction=OrderDirection.CALL,
                duration=120,
            )

    def test_validate_order_parameters_invalid_duration(self, client):
        """Test order parameter validation with invalid duration"""
        with pytest.raises(InvalidParameterError):
            client._validate_order_parameters(
                asset="EURUSD_otc",
                amount=10.0,
                direction=OrderDirection.CALL,
                duration=30,  # Too short
            )

    @pytest.mark.asyncio
    async def test_place_order_success(self, client, mock_websocket):
        """Test successful order placement"""
        with patch.object(client, "_websocket", mock_websocket):
            # Mock websocket as connected
            mock_websocket.websocket = MagicMock()
            mock_websocket.websocket.closed = False
            mock_websocket.connection_info = MagicMock()
            mock_websocket.connection_info.status = ConnectionStatus.CONNECTED

            # Mock order result
            test_order_result = OrderResult(
                order_id="test_order_123",
                asset="EURUSD_otc",
                amount=10.0,
                direction=OrderDirection.CALL,
                duration=120,
                status=OrderStatus.ACTIVE,
                placed_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=120),
            )

            with patch.object(
                client, "_wait_for_order_result", return_value=test_order_result
            ):
                result = await client.place_order(
                    asset="EURUSD_otc",
                    amount=10.0,
                    direction=OrderDirection.CALL,
                    duration=120,
                )

                assert result.order_id == "test_order_123"
                assert result.status == OrderStatus.ACTIVE
                assert result.asset == "EURUSD_otc"

    @pytest.mark.asyncio
    async def test_place_order_not_connected(self, client):
        """Test order placement when not connected"""
        # Mock websocket as not connected
        client._websocket.websocket = None

        with pytest.raises(ConnectionError):
            await client.place_order(
                asset="EURUSD_otc",
                amount=10.0,
                direction=OrderDirection.CALL,
                duration=120,
            )

    @pytest.mark.asyncio
    async def test_get_candles_success(self, client, mock_websocket):
        """Test successful candles retrieval"""
        with patch.object(client, "_websocket", mock_websocket):
            # Mock websocket as connected
            mock_websocket.websocket = MagicMock()
            mock_websocket.websocket.closed = False
            mock_websocket.connection_info = MagicMock()
            mock_websocket.connection_info.status = ConnectionStatus.CONNECTED

            # Mock candles data
            test_candles = [
                {
                    "timestamp": datetime.now(),
                    "open": 1.1000,
                    "high": 1.1010,
                    "low": 1.0990,
                    "close": 1.1005,
                    "asset": "EURUSD_otc",
                    "timeframe": 60,
                }
            ]

            with patch.object(client, "_request_candles", return_value=test_candles):
                candles = await client.get_candles(
                    asset="EURUSD_otc", timeframe="1m", count=100
                )

                assert len(candles) == 1
                assert candles[0]["asset"] == "EURUSD_otc"

    @pytest.mark.asyncio
    async def test_get_candles_invalid_timeframe(self, client):
        """Test candles retrieval with invalid timeframe"""
        # Mock websocket as connected
        client._websocket.websocket = MagicMock()
        client._websocket.websocket.closed = False
        client._websocket.connection_info = MagicMock()
        client._websocket.connection_info.status = ConnectionStatus.CONNECTED

        with pytest.raises(InvalidParameterError):
            await client.get_candles(asset="EURUSD_otc", timeframe="invalid", count=100)

    @pytest.mark.asyncio
    async def test_get_candles_invalid_asset(self, client):
        """Test candles retrieval with invalid asset"""
        # Mock websocket as connected
        client._websocket.websocket = MagicMock()
        client._websocket.websocket.closed = False
        client._websocket.connection_info = MagicMock()
        client._websocket.connection_info.status = ConnectionStatus.CONNECTED

        with pytest.raises(InvalidParameterError):
            await client.get_candles(asset="INVALID_ASSET", timeframe="1m", count=100)

    def test_add_event_callback(self, client):
        """Test adding event callback"""

        def test_callback(data):
            pass

        client.add_event_callback("test_event", test_callback)

        assert "test_event" in client._event_callbacks
        assert test_callback in client._event_callbacks["test_event"]

    def test_remove_event_callback(self, client):
        """Test removing event callback"""

        def test_callback(data):
            pass

        client.add_event_callback("test_event", test_callback)
        client.remove_event_callback("test_event", test_callback)

        assert test_callback not in client._event_callbacks.get("test_event", [])

    @pytest.mark.asyncio
    async def test_context_manager(self, client, mock_websocket):
        """Test async context manager"""
        with patch.object(client, "_websocket", mock_websocket):
            mock_websocket.connect.return_value = True

            async with client:
                assert mock_websocket.connect.called

            mock_websocket.disconnect.assert_called_once()


class TestModels:
    """Test Pydantic models"""

    def test_balance_model(self):
        """Test Balance model"""
        balance = Balance(balance=1000.0, currency="USD", is_demo=True)

        assert balance.balance == 1000.0
        assert balance.currency == "USD"
        assert balance.is_demo is True
        assert isinstance(balance.last_updated, datetime)

    def test_order_model_valid(self):
        """Test Order model with valid data"""
        order = Order(
            asset="EURUSD_otc", amount=10.0, direction=OrderDirection.CALL, duration=120
        )

        assert order.asset == "EURUSD_otc"
        assert order.amount == 10.0
        assert order.direction == OrderDirection.CALL
        assert order.duration == 120
        assert order.request_id is not None

    def test_order_model_invalid_amount(self):
        """Test Order model with invalid amount"""
        with pytest.raises(ValueError):
            Order(
                asset="EURUSD_otc",
                amount=-10.0,  # Negative amount
                direction=OrderDirection.CALL,
                duration=120,
            )

    def test_order_model_invalid_duration(self):
        """Test Order model with invalid duration"""
        with pytest.raises(ValueError):
            Order(
                asset="EURUSD_otc",
                amount=10.0,
                direction=OrderDirection.CALL,
                duration=30,  # Too short
            )

    def test_order_result_model(self):
        """Test OrderResult model"""
        result = OrderResult(
            order_id="test_123",
            asset="EURUSD_otc",
            amount=10.0,
            direction=OrderDirection.CALL,
            duration=120,
            status=OrderStatus.WIN,
            placed_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=120),
            profit=8.0,
        )

        assert result.order_id == "test_123"
        assert result.status == OrderStatus.WIN
        assert result.profit == 8.0


class TestUtilities:
    """Test utility functions"""

    def test_format_session_id(self):
        """Test session ID formatting"""
        from pocketoptionapi_async.utils import format_session_id

        formatted = format_session_id("test_session", True, 123, 1)

        assert "test_session" in formatted
        assert '"isDemo": 1' in formatted
        assert '"uid": 123' in formatted

    def test_calculate_payout_percentage_win(self):
        """Test payout calculation for winning trade"""
        from pocketoptionapi_async.utils import calculate_payout_percentage

        payout = calculate_payout_percentage(1.1000, 1.1010, "call", 0.8)
        assert payout == 0.8

    def test_calculate_payout_percentage_loss(self):
        """Test payout calculation for losing trade"""
        from pocketoptionapi_async.utils import calculate_payout_percentage

        payout = calculate_payout_percentage(1.1000, 1.0990, "call", 0.8)
        assert payout == -1.0


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_async_api.py -v
    pytest.main([__file__, "-v"])
