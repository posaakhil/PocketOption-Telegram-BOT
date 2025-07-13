"""
Test Order Tracking and Logging Fixes
"""

import asyncio
import os
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection


async def test_fixes():
    """Test that order tracking works correctly and logging can be disabled"""

    # Get SSID from environment or use placeholder
    ssid = os.getenv("POCKET_OPTION_SSID", "your_session_id_here")

    if ssid == "your_session_id_here":
        print("Please set POCKET_OPTION_SSID environment variable")
        return

    print("Testing Order Tracking and Logging Fixes...")

    # Test 1: Client with logging enabled (default)
    print("\nTest: Client with logging ENABLED")
    client_with_logs = AsyncPocketOptionClient(ssid, is_demo=True, enable_logging=True)

    try:
        # Connect
        print("📡 Connecting...")
        await client_with_logs.connect()

        if not client_with_logs.is_connected:
            print("Failed to connect")
            return

        print(" Connected successfully")

        # Wait for initialization
        await asyncio.sleep(3)

        # Get balance
        balance = await client_with_logs.get_balance()
        if balance:
            print(f"Balance: ${balance.balance:.2f} (Demo: {balance.is_demo})")

        # Place a test order
        print("\nPlacing test order...")
        order_result = await client_with_logs.place_order(
            asset="EURUSD_otc", amount=1.0, direction=OrderDirection.CALL, duration=60
        )

        print(f"Order placed: {order_result.order_id}")
        print(f"   Status: {order_result.status}")
        print(f"   Error Message: {order_result.error_message or 'None'}")

        # Check if order is properly tracked
        immediate_result = await client_with_logs.check_order_result(
            order_result.order_id
        )
        if immediate_result:
            print(" Order found in tracking system immediately")
        else:
            print("Order NOT found in tracking")

        # Wait a bit to see if it gets resolved
        await asyncio.sleep(10)

        # Check again
        final_result = await client_with_logs.check_order_result(order_result.order_id)
        if final_result:
            print(f"📋 Final order status: {final_result.status}")
            if final_result.profit is not None:
                print(f"Profit: ${final_result.profit:.2f}")

    finally:
        await client_with_logs.disconnect()

    print("\n" + "=" * 50)

    # Test 2: Client with logging disabled
    print("\nTest: Client with logging DISABLED")
    client_no_logs = AsyncPocketOptionClient(ssid, is_demo=True, enable_logging=False)

    try:
        # Connect (should be much quieter)
        print("📡 Connecting (quietly)...")
        await client_no_logs.connect()

        if not client_no_logs.is_connected:
            print("Failed to connect")
            return

        print(" Connected successfully (no logs)")

        # Wait for initialization
        await asyncio.sleep(3)

        # Get balance
        balance = await client_no_logs.get_balance()
        if balance:
            print(f"Balance: ${balance.balance:.2f} (Demo: {balance.is_demo})")

        # Place a test order (should work silently)
        print("\nPlacing test order (silently)...")
        order_result = await client_no_logs.place_order(
            asset="EURUSD_otc", amount=1.0, direction=OrderDirection.CALL, duration=60
        )

        print(f"Order placed: {order_result.order_id}")
        print(f"   Status: {order_result.status}")
        print(f"   Error Message: {order_result.error_message or 'None'}")

        # Check if order is properly tracked
        immediate_result = await client_no_logs.check_order_result(
            order_result.order_id
        )
        if immediate_result:
            print(" Order found in tracking system (silent mode)")
        else:
            print("Order NOT found in tracking")

    finally:
        await client_no_logs.disconnect()

    print("\n Tests completed!")


if __name__ == "__main__":
    # Configure basic logging for test output
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{level}</level> | {message}",
        level="INFO",
    )

    asyncio.run(test_fixes())
