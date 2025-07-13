"""
Complete Demo of Enhanced PocketOption API with Keep-Alive
Demonstrates all the improvements based on the old API patterns
"""

import asyncio
import os
from datetime import datetime
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient


async def demo_enhanced_features():
    """Comprehensive demo of all enhanced features"""

    print("Starting PocketOption Enhanced API Demo")
    print("=" * 60)
    print("Demonstrating all enhancements based on old API patterns:")
    print("Success: Complete SSID format support")
    print("Success: Persistent connections with automatic keep-alive")
    print("Success: Background ping/pong handling (20-second intervals)")
    print("Success: Automatic reconnection with multiple region fallback")
    print("Success: Connection health monitoring and statistics")
    print("Success: Event-driven architecture with callbacks")
    print("Success: Enhanced error handling and recovery")
    print("Success: Modern async/await patterns")
    print("=" * 60)
    print()

    # Complete SSID format (as requested)
    ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'
    print("Authentication: Using complete SSID format:")
    print(f"   {ssid[:80]}...")
    print()

    # Demo 1: Basic Enhanced Client
    print("Demonstration 1: Enhanced Client with Complete SSID")
    print("-" * 50)

    try:
        # Create client with complete SSID (as user requested)
        client = AsyncPocketOptionClient(ssid=ssid, is_demo=True)

        print("Success: Client created with parsed components:")
        print(f"   Session ID: {getattr(client, 'session_id', 'N/A')[:20]}...")
        print(f"   UID: {client.uid}")
        print(f"   Platform: {client.platform}")
        print(f"   Demo Mode: {client.is_demo}")
        print(f"   Fast History: {client.is_fast_history}")

        # Test connection
        print("\nConnecting: Testing connection...")
        try:
            await client.connect()
            if client.is_connected:
                print("Success: Connected successfully!")

                # Show connection stats
                stats = client.get_connection_stats()
                print(f"Statistics: Connection Stats: {stats}")

            else:
                print("Note: Connection failed (expected with test SSID)")
        except Exception as e:
            print(f"Note: Connection error (expected): {str(e)[:100]}...")

        await client.disconnect()

    except Exception as e:
        print(f"Note: Client demonstration error: {e}")

    print()

    # Demo 2: Persistent Connection Features
    print("Persistent: Demonstration 2: Persistent Connection with Keep-Alive")
    print("-" * 50)

    try:
        # Create client with persistent connection enabled
        persistent_client = AsyncPocketOptionClient(
            ssid=ssid,
            is_demo=True,
            persistent_connection=True,  # Enable keep-alive like old API
            auto_reconnect=True,
        )

        # Add event handlers to monitor connection events
        events_log = []

        def on_connected(data):
            events_log.append(
                f"CONNECTED: {datetime.now().strftime('%H:%M:%S')} - {data}"
            )
            print(
                f"Successfully: Event: Connected at {datetime.now().strftime('%H:%M:%S')}"
            )

        def on_reconnected(data):
            events_log.append(
                f"RECONNECTED: {datetime.now().strftime('%H:%M:%S')} - {data}"
            )
            print(
                f"Reconnection: Event: Reconnected at {datetime.now().strftime('%H:%M:%S')}"
            )

        def on_authenticated(data):
            events_log.append(f"AUTHENTICATED: {datetime.now().strftime('%H:%M:%S')}")
            print(
                f"Success: Event: Authenticated at {datetime.now().strftime('%H:%M:%S')}"
            )

        persistent_client.add_event_callback("connected", on_connected)
        persistent_client.add_event_callback("reconnected", on_reconnected)
        persistent_client.add_event_callback("authenticated", on_authenticated)

        print("Starting persistent connection...")
        try:
            success = await persistent_client.connect(persistent=True)

            if success:
                print("Success: Persistent connection established")

                # Monitor for 30 seconds to show keep-alive behavior
                print("Statistics: Monitoring persistent connection (30 seconds)...")
                print("   Watch for automatic pings and reconnection attempts...")

                for i in range(30):
                    await asyncio.sleep(1)

                    # Show stats every 10 seconds
                    if i % 10 == 0 and i > 0:
                        stats = persistent_client.get_connection_stats()
                        print(
                            f"   Data: [{i}s] Connected: {persistent_client.is_connected}, "
                            f"Messages sent: {stats.get('messages_sent', 0)}, "
                            f"Reconnects: {stats.get('total_reconnects', 0)}"
                        )

                # Show final event log
                print(f"\nDemonstration: Connection Events ({len(events_log)} total):")
                for event in events_log:
                    print(f"   • {event}")

            else:
                print("Note: Persistent connection failed (expected with test SSID)")

        except Exception as e:
            print(f"Note: Persistent connection error: {str(e)[:100]}...")

        await persistent_client.disconnect()

    except Exception as e:
        print(f"Note: Persistent demonstration error: {e}")

    print()

    # Demo 3: API Features with Real Data (if available)
    print("Statistics: Demonstration 3: API Features and Data Operations")
    print("-" * 50)

    real_ssid = os.getenv("POCKET_OPTION_SSID")
    if real_ssid and "n1p5ah5u8t9438rbunpgrq0hlq" not in real_ssid:
        print("Authentication: Real SSID detected - testing with live connection...")

        try:
            live_client = AsyncPocketOptionClient(
                ssid=real_ssid, is_demo=True, auto_reconnect=True
            )

            success = await live_client.connect()
            if success:
                print("Success: Live connection established")

                # Test balance
                try:
                    balance = await live_client.get_balance()
                    print(f"Balance: ${balance.balance:.2f} {balance.currency}")
                except Exception as e:
                    print(f"Note: Balance test: {e}")

                # Test candles
                try:
                    candles = await live_client.get_candles("EURUSD_otc", "1m", 5)
                    print(f"Retrieved: Retrieved {len(candles)} candles for EURUSD_otc")

                    # Test DataFrame conversion
                    df = await live_client.get_candles_dataframe("EURUSD_otc", "1m", 5)
                    print(f"Statistics: DataFrame shape: {df.shape}")
                except Exception as e:
                    print(f"Note: Candles test: {e}")

                # Test health monitoring
                health = await live_client.get_health_status()
                print(f"Health Status: {health}")

                # Test performance metrics
                metrics = await live_client.get_performance_metrics()
                print(f"Statistics: Performance Metrics: {metrics}")

            await live_client.disconnect()

        except Exception as e:
            print(f"Error: Live demonstration error: {e}")
    else:
        print("Note: Skipping live demo - requires real SSID")
        print(
            "   Set environment variable: export POCKET_OPTION_SSID='your_complete_ssid'"
        )

    print()


def show_api_improvements():
    """Show comparison with old API"""

    print("Analysis: API Improvements Summary")
    print("=" * 60)

    print("Architecture: ARCHITECTURE IMPROVEMENTS:")
    print("   Old API: Synchronous with threading")
    print("   New API: Fully async/await with modern patterns")
    print()

    print("Connection: CONNECTION MANAGEMENT:")
    print("   Old API: Manual daemon threads + run_forever()")
    print("   New API: Persistent connections with asyncio tasks")
    print("           Success: Automatic ping every 20 seconds")
    print("           Success: Health monitoring and statistics")
    print("           Success: Graceful reconnection handling")
    print()

    print("Message: MESSAGE HANDLING:")
    print("   Old API: Basic message processing")
    print("   New API: Optimized message routing with caching")
    print("           Success: Message batching for performance")
    print("           Success: Event-driven callbacks")
    print("           Success: Type-safe message models")
    print()

    print("Error Handling: ERROR HANDLING:")
    print("   Old API: Basic try/catch with global variables")
    print("   New API: Comprehensive error monitoring")
    print("           Success: Circuit breaker pattern")
    print("           Success: Retry mechanisms with backoff")
    print("           Success: Health checks and alerting")
    print()

    print("Statistics: DATA MANAGEMENT:")
    print("   Old API: Basic data structures")
    print("   New API: Modern data handling")
    print("           Success: Pydantic models for type safety")
    print("           Success: pandas DataFrame integration")
    print("           Success: Automatic data validation")
    print()

    print("Developer Experience: DEVELOPER EXPERIENCE:")
    print("   Old API: Manual setup and configuration")
    print("   New API: Enhanced developer tools")
    print("           Success: Rich logging with loguru")
    print("           Success: Context manager support")
    print("           Success: Comprehensive testing")
    print("           Success: Performance monitoring")
    print()

    print("Usage Examples: USAGE EXAMPLES:")
    print()

    print("   OLD API STYLE:")
    print("   ```python")
    print("   api = PocketOption(ssid, demo=True)")
    print("   api.connect()")
    print("   while api.check_connect():")
    print("       balance = api.get_balance()")
    print("       time.sleep(1)")
    print("   ```")
    print()

    print("   NEW API STYLE:")
    print("   ```python")
    print('   ssid = r\'42["auth",{"session":"...","isDemo":1,"uid":123}]\'')
    print(
        "   async with AsyncPocketOptionClient(ssid=ssid, persistent_connection=True) as client:"
    )
    print("       balance = await client.get_balance()")
    print("       df = await client.get_candles_dataframe('EURUSD_otc', '1m', 100)")
    print("       # Connection maintained automatically with keep-alive")
    print("   ```")
    print()


def show_keep_alive_features():
    """Show specific keep-alive features"""

    print("Persistent: Keep-Alive Features Based on Old API Analysis")
    print("=" * 60)

    print("Demonstration: IMPLEMENTED FEATURES:")
    print("Success: Continuous ping loop (20-second intervals)")
    print("Success: Automatic reconnection on disconnection")
    print("Success: Multiple region fallback")
    print("Success: Background task management")
    print("Success: Connection health monitoring")
    print("Success: Message routing and processing")
    print("Success: Event-driven callbacks")
    print("Success: Connection statistics tracking")
    print("Success: Graceful shutdown and cleanup")
    print("Success: Complete SSID format support")
    print()

    print("Technical Implementation: TECHNICAL IMPLEMENTATION:")
    print("• AsyncWebSocketClient with persistent connections")
    print("• ConnectionKeepAlive manager for advanced scenarios")
    print("• Background asyncio tasks for ping/reconnect")
    print("• Event handlers for connection state changes")
    print("• Connection pooling with performance stats")
    print("• Message batching and optimization")
    print("• Health monitoring with alerts")
    print()

    print("Statistics: MONITORING CAPABILITIES:")
    print("• Connection uptime tracking")
    print("• Message send/receive counters")
    print("• Reconnection attempt statistics")
    print("• Ping response time monitoring")
    print("• Health check results")
    print("• Performance metrics collection")
    print()

    print("Configuration: CONFIGURATION OPTIONS:")
    print("• persistent_connection: Enable advanced keep-alive")
    print("• auto_reconnect: Automatic reconnection on failure")
    print("• ping_interval: Customizable ping frequency")
    print("• max_reconnect_attempts: Reconnection retry limit")
    print("• connection_timeout: Connection establishment timeout")
    print("• health_check_interval: Health monitoring frequency")


async def main():
    """Main demo function"""

    logger.info("Starting Enhanced PocketOption API Demo")

    # Run comprehensive demo
    await demo_enhanced_features()

    # Show improvements
    show_api_improvements()

    # Show keep-alive features
    show_keep_alive_features()

    print()
    print("Enhanced PocketOption API Demo Complete!")
    print()
    print("Next Steps:")
    print("1. Set your real SSID: export POCKET_OPTION_SSID='your_complete_ssid'")
    print("2. Use persistent_connection=True for long-running applications")
    print("3. Monitor connection with get_connection_stats()")
    print("4. Add event callbacks for connection management")
    print("5. Use async context managers for automatic cleanup")
    print()
    print("Documentation: README_ASYNC.md")
    print("Examples: examples/async_examples.py")
    print("Tests: test_persistent_connection.py")


if __name__ == "__main__":
    asyncio.run(main())
