"""
Complete Order Tracking Test - Final Version
Tests all the fixes made to the order tracking system:
1. Order placement without duplication
2. Proper waiting for server responses
3. Event-driven order completion tracking
4. Fallback handling for timeouts
"""

import asyncio
import os
from datetime import datetime, timedelta
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection


async def test_complete_order_lifecycle():
    """Test the complete order lifecycle with all fixes"""

    # Get SSID from environment
    ssid = os.getenv("POCKET_OPTION_SSID")

    if not ssid:
        print("Please set POCKET_OPTION_SSID environment variable")
        print("Example: set POCKET_OPTION_SSID='your_session_id_here'")
        return

    print("Complete Order Tracking Test - Final Version")
    print("=" * 60)

    # Create client
    client = AsyncPocketOptionClient(ssid, is_demo=True)

    try:
        # Connect
        print("📡 Connecting...")
        await client.connect()

        if not client.is_connected:
            print("Failed to connect")
            return

        print(" Connected successfully")

        # Wait for initialization
        await asyncio.sleep(3)

        # Get balance
        balance = await client.get_balance()
        if balance:
            print(f"Balance: ${balance.balance:.2f} (Demo: {balance.is_demo})")
        else:
            print("No balance received")

        # Test 1: Order Placement (should not create duplicates)
        print("\n📋 TEST 1: Order Placement Without Duplication")
        print("-" * 50)

        # Check initial active orders count
        initial_active = await client.get_active_orders()
        print(f"📊 Initial active orders: {len(initial_active)}")

        # Place order
        print("Placing order...")
        order_result = await client.place_order(
            asset="EURUSD_otc",
            amount=1.0,
            direction=OrderDirection.CALL,
            duration=60,  # 1 minute
        )

        print(f" Order placed: {order_result.order_id}")
        print(f"   Status: {order_result.status}")
        print(f"   Asset: {order_result.asset}")
        print(f"   Amount: ${order_result.amount}")
        print(f"   Direction: {order_result.direction}")
        print(f"   Duration: {order_result.duration}s")

        # Test 2: No Duplication Check
        print("\n📋 TEST 2: No Order Duplication Check")
        print("-" * 50)

        # Check that only one order was created
        active_orders_after = await client.get_active_orders()
        added_orders = len(active_orders_after) - len(initial_active)

        if added_orders == 1:
            print(" PASS: Exactly 1 order was created (no duplication)")
        else:
            print(f"FAIL: {added_orders} orders were created (expected 1)")
            for order in active_orders_after:
                print(f"   - {order.order_id}: {order.status}")

        # Test 3: Order Tracking
        print("\n📋 TEST 3: Order Tracking and Result Checking")
        print("-" * 50)

        # Immediate check
        immediate_result = await client.check_order_result(order_result.order_id)
        if immediate_result:
            print(" Order immediately found in tracking system")
            print(f"   ID: {immediate_result.order_id}")
            print(f"   Status: {immediate_result.status}")
        else:
            print("Order NOT found in tracking system - this is a problem!")
            return

        # Test 4: Event-Based Order Completion Monitoring
        print("\n📋 TEST 4: Event-Based Order Completion")
        print("-" * 50)

        # Set up event callback to detect completion
        completed_orders = []

        def on_order_closed(order_result):
            completed_orders.append(order_result)
            status = (
                "WIN"
                if order_result.profit > 0
                else "LOSE"
                if order_result.profit < 0
                else "EVEN"
            )
            print(
                f"ORDER COMPLETED via EVENT: {status} - Profit: ${order_result.profit:.2f}"
            )

        client.add_event_callback("order_closed", on_order_closed)

        # Test 5: Wait for Trade Completion
        print("\n📋 TEST 5: Waiting for Trade Completion")
        print("-" * 50)

        print(
            f"⏱️ Waiting for trade to complete (up to {order_result.duration + 30} seconds)..."
        )
        start_time = datetime.now()
        max_wait = timedelta(
            seconds=order_result.duration + 30
        )  # Trade duration + buffer

        last_status = None

        while datetime.now() - start_time < max_wait:
            result = await client.check_order_result(order_result.order_id)

            if result:
                # Only print status changes to avoid spam
                if result.status != last_status:
                    status_emoji = (
                        "🟢"
                        if result.status == "active"
                        else "🔴"
                        if result.status in ["win", "lose"]
                        else "🟡"
                    )
                    print(f"   {status_emoji} Order status: {result.status}")
                    last_status = result.status

                # Check if order completed
                if result.profit is not None:
                    win_lose = (
                        "WIN"
                        if result.profit > 0
                        else "LOSE"
                        if result.profit < 0
                        else "EVEN"
                    )
                    print("\nTRADE COMPLETED!")
                    print(f"   Result: {win_lose}")
                    print(f"   Profit/Loss: ${result.profit:.2f}")
                    if result.payout:
                        print(f"   Payout: ${result.payout:.2f}")

                    # Calculate percentage return
                    if result.profit != 0:
                        percentage = (result.profit / order_result.amount) * 100
                        print(f"   Return: {percentage:.1f}%")

                    break

                # Check if status indicates completion but no profit yet
                elif result.status in ["win", "lose", "closed"]:
                    print(
                        f"   📊 Order marked as {result.status} but no profit data yet..."
                    )

            else:
                print("   Order disappeared from tracking system")
                break

            await asyncio.sleep(2)  # Check every 2 seconds

        # Test 6: Event vs Polling Comparison
        print("\n📋 TEST 6: Event vs Polling Results")
        print("-" * 50)

        # Check if we completed via event callback
        if completed_orders:
            print(" Order completion detected via EVENT callback!")
            final_order_event = completed_orders[0]
            print(f"   Event Result - Profit: ${final_order_event.profit:.2f}")
        else:
            print("No completion event received")

        # Check final polling result
        final_result_poll = await client.check_order_result(order_result.order_id)
        if final_result_poll:
            print(" Order completion detected via POLLING!")
            print(
                f"   Polling Result - Profit: ${final_result_poll.profit:.2f if final_result_poll.profit is not None else 'None'}"
            )
        else:
            print("Order not found via polling")

        # Test 7: Final System State
        print("\n📋 TEST 7: Final System State")
        print("-" * 50)

        # Check final counts
        final_active_orders = await client.get_active_orders()
        print(f"📊 Final active orders: {len(final_active_orders)}")

        for order in final_active_orders:
            print(f"   Active: {order.order_id} - {order.status}")

        # Show test summary
        print("\n📋 TEST SUMMARY")
        print("=" * 60)

        tests_passed = 0
        total_tests = 7

        # Test results
        if added_orders == 1:
            print(" Order Placement (No Duplication): PASS")
            tests_passed += 1
        else:
            print("Order Placement (No Duplication): FAIL")

        if immediate_result:
            print(" Order Tracking: PASS")
            tests_passed += 1
        else:
            print("Order Tracking: FAIL")

        if completed_orders:
            print(" Event-Based Completion: PASS")
            tests_passed += 1
        else:
            print("Event-Based Completion: FAIL")

        if final_result_poll and final_result_poll.profit is not None:
            print(" Polling-Based Completion: PASS")
            tests_passed += 1
        else:
            print("Polling-Based Completion: FAIL")

        # Additional checks
        if len(final_active_orders) < len(active_orders_after):
            print(" Order Movement (Active -> Completed): PASS")
            tests_passed += 1
        else:
            print("Order Movement (Active -> Completed): FAIL")

        if balance:
            print(" Balance Retrieval: PASS")
            tests_passed += 1
        else:
            print("Balance Retrieval: FAIL")

        print(f"\nOVERALL RESULT: {tests_passed}/{total_tests} tests passed")

        if tests_passed >= 5:
            print("🎉 ORDER TRACKING SYSTEM IS WORKING WELL!")
        elif tests_passed >= 3:
            print("Order tracking is partially working, some improvements needed")
        else:
            print("Major issues with order tracking system")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Disconnect
        print("\nDisconnecting...")
        await client.disconnect()
        print(" Test completed")


if __name__ == "__main__":
    # Configure logging to be less verbose
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{level}</level> | {message}",
        level="ERROR",  # Only show errors from the library to keep output clean
    )

    asyncio.run(test_complete_order_lifecycle())
