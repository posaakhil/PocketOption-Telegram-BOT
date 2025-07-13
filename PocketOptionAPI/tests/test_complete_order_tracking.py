"""
Complete Order Tracking Test
Tests the full order lifecycle including waiting for trade completion and profit/loss tracking
"""

import asyncio
import os
from datetime import datetime, timedelta
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection


async def wait_for_trade_completion():
    """Test complete order lifecycle with profit tracking"""

    # Get SSID from environment
    ssid = os.getenv("POCKET_OPTION_SSID")

    if not ssid:
        print("Please set POCKET_OPTION_SSID environment variable")
        print("Example: set POCKET_OPTION_SSID='your_session_id_here'")
        return

    print("Complete Order Tracking Test")
    print("=" * 50)

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

        # Add event callback to monitor order completion
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
            print(f"Order completed: {status} - Profit: ${order_result.profit:.2f}")

        client.add_event_callback("order_closed", on_order_closed)

        # Place a test order with shorter duration for faster results
        print("\nPlacing test order...")
        order_result = await client.place_order(
            asset="EURUSD_otc",
            amount=1.0,
            direction=OrderDirection.CALL,
            duration=60,  # 1 minute for quick testing
        )

        print(f" Order placed: {order_result.order_id}")
        print(f"   Status: {order_result.status}")
        print(f"   Asset: {order_result.asset}")
        print(f"   Amount: ${order_result.amount}")
        print(f"   Direction: {order_result.direction}")
        print(f"   Duration: {order_result.duration}s")
        print(f"   Expires at: {order_result.expires_at.strftime('%H:%M:%S')}")

        # Check immediate order result
        immediate_result = await client.check_order_result(order_result.order_id)
        if immediate_result:
            print(" Order immediately found in tracking system")
        else:
            print("Order NOT found in tracking system - this is a problem!")
            return

        # Wait for the trade to complete
        print(
            f"\n⏱️ Waiting for trade to complete (up to {order_result.duration + 30} seconds)..."
        )
        start_time = datetime.now()
        max_wait = timedelta(
            seconds=order_result.duration + 30
        )  # Trade duration + 30 seconds buffer

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

        # Check if we completed via event callback
        if completed_orders:
            print("\n Order completion detected via event callback!")
            final_order = completed_orders[0]
            print(f"   Final profit: ${final_order.profit:.2f}")

        # Final status check
        final_result = await client.check_order_result(order_result.order_id)
        if final_result:
            print("\n📋 Final status:")
            print(f"   Order ID: {final_result.order_id}")
            print(f"   Status: {final_result.status}")
            if final_result.profit is not None:
                print(f"   Final Profit/Loss: ${final_result.profit:.2f}")
            else:
                print("   No profit data available (may indicate tracking issue)")
        else:
            print("\nCould not find final order result")

        # Show active orders count
        active_orders = await client.get_active_orders()
        print(f"\n📊 Active orders remaining: {len(active_orders)}")

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
        level="WARNING",  # Only show warnings and errors from the library
    )

    asyncio.run(wait_for_trade_completion())
