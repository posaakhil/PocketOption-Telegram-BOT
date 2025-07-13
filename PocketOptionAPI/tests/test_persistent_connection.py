"""
Test script for persistent connection with keep-alive functionality
Demonstrates the enhanced connection management based on old API patterns
"""

import asyncio
import os
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient


async def test_persistent_connection():
    """Test persistent connection with automatic keep-alive"""

    print("Testing Persistent Connection with Keep-Alive")
    print("=" * 60)
    print("This test demonstrates the enhanced connection management")
    print("based on the old API's proven keep-alive patterns:")
    print(" Automatic ping every 20 seconds")
    print(" Automatic reconnection on disconnection")
    print(" Multiple region fallback")
    print(" Background task management")
    print(" Connection health monitoring")
    print("=" * 60)
    print()

    # Complete SSID format
    complete_ssid = os.getenv(
        "POCKET_OPTION_SSID",
        r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":0,"platform":1}]',
    )

    if "n1p5ah5u8t9438rbunpgrq0hlq" in complete_ssid:
        print(
            " Using test SSID - connection will fail but demonstrates the keep-alive logic"
        )
        print(
            "   For real testing, set: export POCKET_OPTION_SSID='your_complete_ssid'"
        )
        print()

    # Test 1: Regular connection (existing behavior)
    print("Test 1: Regular Connection (with basic keep-alive)")
    print("-" * 50)

    try:
        client_regular = AsyncPocketOptionClient(
            ssid=complete_ssid,
            is_demo=True,
            persistent_connection=False,  # Regular connection
            auto_reconnect=True,  # But with auto-reconnect
        )

        print("📊 Connecting with regular mode...")
        success = await client_regular.connect()

        if success:
            print(" Regular connection established")

            # Monitor for 30 seconds
            print("📊 Monitoring regular connection for 30 seconds...")
            for i in range(30):
                await asyncio.sleep(1)

                if i % 10 == 0:
                    stats = client_regular.get_connection_stats()
                    print(
                        f"   Stats: Connected={client_regular.is_connected}, "
                        f"Pings sent={stats.get('messages_sent', 0)}, "
                        f"Reconnects={stats.get('total_reconnects', 0)}"
                    )
        else:
            print(" Regular connection failed (expected with test SSID)")

        await client_regular.disconnect()
        print(" Regular connection test completed")

    except Exception as e:
        print(f" Regular connection error (expected): {str(e)[:100]}...")

    print()

    # Test 2: Persistent connection (new enhanced behavior)
    print("Test 2: Persistent Connection (enhanced keep-alive)")
    print("-" * 50)

    try:
        client_persistent = AsyncPocketOptionClient(
            ssid=complete_ssid,
            is_demo=True,
            persistent_connection=True,  # Enhanced persistent mode
            auto_reconnect=True,
        )

        # Add event handlers to monitor keep-alive events
        connection_events = []

        def on_connected(data):
            connection_events.append(f"Connected: {data}")
            print(f"🎉 Event: Connected to {data}")

        def on_reconnected(data):
            connection_events.append(f"Reconnected: {data}")
            print(f"Event: Reconnected after {data}")

        def on_authenticated(data):
            connection_events.append(f"Authenticated: {data}")
            print(" Event: Authenticated")

        client_persistent.add_event_callback("connected", on_connected)
        client_persistent.add_event_callback("reconnected", on_reconnected)
        client_persistent.add_event_callback("authenticated", on_authenticated)

        print("📊 Connecting with persistent mode...")
        success = await client_persistent.connect()

        if success:
            print(" Persistent connection established with keep-alive active")

            # Monitor for 60 seconds to see keep-alive in action
            print("📊 Monitoring persistent connection for 60 seconds...")
            print("   (Watch for automatic pings every 20 seconds)")

            for i in range(60):
                await asyncio.sleep(1)

                # Print stats every 15 seconds
                if i % 15 == 0 and i > 0:
                    stats = client_persistent.get_connection_stats()
                    print(
                        f"   Stats: Connected={client_persistent.is_connected}, "
                        f"Pings={stats.get('last_ping_time')}, "
                        f"Messages sent={stats.get('messages_sent', 0)}, "
                        f"Messages received={stats.get('messages_received', 0)}, "
                        f"Reconnects={stats.get('total_reconnects', 0)}, "
                        f"Uptime={stats.get('uptime', 'N/A')}"
                    )

                # Send test message every 30 seconds
                if i % 30 == 0 and i > 0:
                    print("   📤 Sending test message...")
                    await client_persistent.send_message('42["test"]')

            # Show final statistics
            final_stats = client_persistent.get_connection_stats()
            print("\n📊 Final Connection Statistics:")
            print(f"   Total connections: {final_stats.get('total_connections', 0)}")
            print(
                f"   Successful connections: {final_stats.get('successful_connections', 0)}"
            )
            print(f"   Total reconnects: {final_stats.get('total_reconnects', 0)}")
            print(f"   Messages sent: {final_stats.get('messages_sent', 0)}")
            print(f"   Messages received: {final_stats.get('messages_received', 0)}")
            print(f"   Connection uptime: {final_stats.get('uptime', 'N/A')}")
            print(f"   Last ping: {final_stats.get('last_ping_time', 'None')}")
            print(f"   Available regions: {final_stats.get('available_regions', 0)}")

            print(f"\n📋 Connection Events ({len(connection_events)} total):")
            for event in connection_events[-5:]:  # Show last 5 events
                print(f"   • {event}")

        else:
            print(" Persistent connection failed (expected with test SSID)")

        await client_persistent.disconnect()
        print(" Persistent connection test completed")

    except Exception as e:
        print(f" Persistent connection error (expected): {str(e)[:100]}...")

    print()

    # Test 3: Connection resilience simulation
    print("Test 3: Connection Resilience Simulation")
    print("-" * 50)
    print("This would test automatic reconnection when connection drops")
    print("(Requires real SSID for full testing)")

    real_ssid = os.getenv("POCKET_OPTION_SSID")
    if real_ssid and "n1p5ah5u8t9438rbunpgrq0hlq" not in real_ssid:
        print("🔑 Real SSID detected, testing with actual connection...")

        try:
            resilience_client = AsyncPocketOptionClient(
                ssid=real_ssid,
                is_demo=True,
                persistent_connection=True,
                auto_reconnect=True,
            )

            print("📊 Establishing resilient connection...")
            success = await resilience_client.connect()

            if success:
                print(" Resilient connection established")

                # Monitor for 2 minutes
                print("📊 Monitoring resilient connection for 2 minutes...")
                for i in range(120):
                    await asyncio.sleep(1)

                    if i % 30 == 0:
                        stats = resilience_client.get_connection_stats()
                        print(
                            f"   Stats: Connected={resilience_client.is_connected}, "
                            f"Uptime={stats.get('uptime', 'N/A')}"
                        )

                        # Try to get balance to test API functionality
                        try:
                            balance = await resilience_client.get_balance()
                            print(f"   Balance: ${balance.balance:.2f}")
                        except Exception as e:
                            print(f"    Balance check failed: {e}")

                await resilience_client.disconnect()
                print(" Resilience test completed")
            else:
                print("Resilient connection failed")

        except Exception as e:
            print(f"Resilience test error: {e}")
    else:
        print(" Skipping resilience test (requires real SSID)")

    print()
    print("🎉 All persistent connection tests completed!")
    print()
    print("📋 Summary of Enhanced Features:")
    print(" Persistent connections with automatic keep-alive")
    print(" Automatic reconnection with multiple region fallback")
    print(" Background ping/pong handling (20-second intervals)")
    print(" Connection health monitoring and statistics")
    print(" Event-driven connection management")
    print(" Graceful connection cleanup and resource management")
    print()
    print("💡 Usage Tips:")
    print("• Use persistent_connection=True for long-running applications")
    print("• Set auto_reconnect=True for automatic recovery from disconnections")
    print("• Monitor connection statistics with get_connection_stats()")
    print("• Add event callbacks to handle connection events")


async def test_comparison_with_old_api():
    """Compare new API behavior with old API patterns"""

    print("\n🔍 Comparison with Old API Patterns")
    print("=" * 50)

    print("Old API Features → New Async API Implementation:")
    print("• daemon threads → asyncio background tasks")
    print("• ping every 20s → async ping loop with '42[\"ps\"]'")
    print("• auto reconnect → enhanced reconnection monitor")
    print("• global_value tracking → connection statistics")
    print("• websocket.run_forever() → persistent connection manager")
    print("• manual error handling → automatic exception recovery")
    print("• blocking operations → non-blocking async operations")
    print()

    print("Enhanced Features in New API:")
    print("✨ Type safety with Pydantic models")
    print("✨ Comprehensive error monitoring and health checks")
    print("✨ Event-driven architecture with callbacks")
    print("✨ Connection pooling and performance optimization")
    print("✨ Graceful shutdown and resource cleanup")
    print("✨ Modern async/await patterns")
    print("✨ Built-in rate limiting and message batching")
    print("✨ pandas DataFrame integration")
    print("✨ Rich logging and debugging information")


if __name__ == "__main__":
    logger.info("Testing Enhanced Persistent Connection Functionality")

    # Run tests
    asyncio.run(test_persistent_connection())
    asyncio.run(test_comparison_with_old_api())
