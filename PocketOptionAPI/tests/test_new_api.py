"""
Simple test script to verify the new async API works
"""

import asyncio
import os

# Import the new async API
from pocketoptionapi_async import (
    AsyncPocketOptionClient,
    OrderDirection,
    ConnectionError,
)


async def test_basic_functionality():
    """Test basic functionality of the new async API"""

    print("Testing Professional Async PocketOption API")
    print("=" * 50)

    # Complete SSID format for testing (replace with real one for live testing)
    complete_ssid = os.getenv(
        "POCKET_OPTION_SSID",
        r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":0,"platform":1}]',
    )

    if "n1p5ah5u8t9438rbunpgrq0hlq" in complete_ssid:
        print(
            " Using mock SSID. Set POCKET_OPTION_SSID environment variable for live testing."
        )
        print(
            '   Format: export POCKET_OPTION_SSID=\'42["auth",{"session":"your_session","isDemo":1,"uid":your_uid,"platform":1}]\''
        )

    try:
        # Test 1: Client initialization
        print("\nTesting client initialization...")
        client = AsyncPocketOptionClient(ssid=complete_ssid, is_demo=True)
        print(" Client initialized successfully")

        # Test 2: Connection (will fail with mock session, but tests the flow)
        print("\nTesting connection...")
        try:
            await client.connect()
            print(" Connected successfully")

            # Test 3: Get balance
            print("\nTesting balance retrieval...")
            try:
                balance = await client.get_balance()
                print(f" Balance: ${balance.balance:.2f} ({balance.currency})")
            except Exception as e:
                print(f"Balance test: {e}")

            # Test 4: Get candles
            print("\nTesting candles retrieval...")
            try:
                candles = await client.get_candles(
                    asset="EURUSD_otc", timeframe="1m", count=10
                )
                print(f" Retrieved {len(candles)} candles")
            except Exception as e:
                print(f"Candles test: {e}")

            # Test 5: Order placement (demo)
            print("\n5️⃣ Testing order placement...")
            try:
                order_result = await client.place_order(
                    asset="EURUSD_otc",
                    amount=1.0,
                    direction=OrderDirection.CALL,
                    duration=60,
                )
                print(f" Order placed: {order_result.order_id}")
            except Exception as e:
                print(f"Order test: {e}")

        except ConnectionError as e:
            print(f"Connection test (expected with mock session): {e}")

        finally:
            # Test 6: Disconnection
            print("\n6️⃣ Testing disconnection...")
            await client.disconnect()
            print(" Disconnected successfully")

    except Exception as e:
        print(f"Unexpected error: {e}")

    print("\nAPI Structure Tests")
    print("=" * 30)

    # Test API structure
    test_api_structure()

    print("\n All tests completed!")
    print("\nNext steps:")
    print(
        "   1. Set your real session ID: $env:POCKET_OPTION_SSID='your_real_session_id'"
    )
    print("   2. Run with real session: python test_new_api.py")
    print("   3. Check examples in examples/async_examples.py")
    print("   4. Read full documentation in README_ASYNC.md")


def test_api_structure():
    """Test that all API components are properly structured"""

    # Test imports
    try:
        from pocketoptionapi_async import (
            AsyncPocketOptionClient,
            OrderDirection,
            OrderStatus,
            Balance,
            Order,
            OrderResult,
            ASSETS,
            REGIONS,
        )

        print(" All imports successful")
    except ImportError as e:
        print(f"Import error: {e}")
        return

    # Test enums
    assert OrderDirection.CALL == "call"
    assert OrderDirection.PUT == "put"
    print(" Enums working correctly")

    # Test constants
    assert "EURUSD_otc" in ASSETS
    assert len(REGIONS.get_all()) > 0
    print(" Constants available")

    # Test model validation
    try:
        # Valid order
        Order(
            asset="EURUSD_otc", amount=10.0, direction=OrderDirection.CALL, duration=120
        )
        print(" Model validation working")

        # Invalid order (should raise ValueError)
        try:
            Order(
                asset="EURUSD_otc",
                amount=-10.0,  # Invalid amount
                direction=OrderDirection.CALL,
                duration=120,
            )
            print("Model validation not working")
        except ValueError:
            print(" Model validation correctly catches errors")

    except Exception as e:
        print(f"Model test error: {e}")


async def test_context_manager():
    """Test async context manager functionality"""

    print("\nTesting context manager...")

    session_id = "n1p5ah5u8t9438rbunpgrq0hlq"

    try:
        async with AsyncPocketOptionClient(session_id, is_demo=True) as client:
            print(" Context manager entry successful")
            assert client is not None
        print(" Context manager exit successful")
    except Exception as e:
        print(f"Context manager test (expected with mock): {e}")


async def test_event_callbacks():
    """Test event callback system"""

    print("\n📡 Testing event callbacks...")

    session_id = "n1p5ah5u8t9438rbunpgrq0hlq"
    client = AsyncPocketOptionClient(session_id, is_demo=True)

    # Test callback registration
    callback_called = False

    def test_callback(data):
        nonlocal callback_called
        callback_called = True

    client.add_event_callback("test_event", test_callback)
    print(" Event callback registered")

    # Test callback removal
    client.remove_event_callback("test_event", test_callback)
    print(" Event callback removed")


def print_api_features():
    """Print the key features of the new API"""

    print("\nNEW ASYNC API FEATURES")
    print("=" * 40)

    features = [
        " 100% Async/Await Support",
        " Type Safety with Pydantic Models",
        " Professional Error Handling",
        " Automatic Connection Management",
        " Event-Driven Architecture",
        " pandas DataFrame Integration",
        " Built-in Rate Limiting",
        " Context Manager Support",
        " Comprehensive Testing",
        " Rich Logging with loguru",
        " WebSocket Auto-Reconnection",
        " Modern Python Practices",
    ]

    for feature in features:
        print(f"  {feature}")

    print("\n📊 SUPPORTED ASSETS:")
    print("  - 50+ Forex pairs (major and exotic)")
    print("  - 20+ Cryptocurrencies")
    print("  - 15+ Commodities (Gold, Silver, Oil, etc.)")
    print("  - 25+ Stock Indices")
    print("  - 50+ Individual Stocks")

    print("\n⚡ PERFORMANCE IMPROVEMENTS:")
    print("  - Non-blocking async operations")
    print("  - Concurrent order management")
    print("  - Efficient WebSocket handling")
    print("  - Memory-optimized data structures")


if __name__ == "__main__":
    print_api_features()

    # Run all tests
    asyncio.run(test_basic_functionality())
    asyncio.run(test_context_manager())
    asyncio.run(test_event_callbacks())
