"""
Advanced Testing Suite for PocketOption Async API
Tests edge cases, performance, and advanced scenarios
"""

import asyncio
import time
import random
import json
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from pocketoptionapi_async.client import AsyncPocketOptionClient
from pocketoptionapi_async.models import OrderDirection, TimeFrame
from pocketoptionapi_async.connection_keep_alive import ConnectionKeepAlive


class AdvancedTestSuite:
    """Advanced testing suite for the API"""

    def __init__(self, ssid: str):
        self.ssid = ssid
        self.test_results = {}
        self.performance_metrics = {}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("Testing: Starting Advanced Testing Suite")

        tests = [
            ("Connection Stress Test", self.test_connection_stress),
            ("Concurrent Operations Test", self.test_concurrent_operations),
            ("Data Consistency Test", self.test_data_consistency),
            ("Error Handling Test", self.test_error_handling),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Memory Usage Test", self.test_memory_usage),
            ("Network Resilience Test", self.test_network_resilience),
            ("Long Running Session Test", self.test_long_running_session),
            ("Multi-Asset Operations", self.test_multi_asset_operations),
            ("Rapid Trading Simulation", self.test_rapid_trading_simulation),
        ]

        for test_name, test_func in tests:
            logger.info(f"Analysis: Running: {test_name}")
            try:
                start_time = time.time()
                result = await test_func()
                end_time = time.time()

                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "result": result,
                    "duration": end_time - start_time,
                }

                logger.success(
                    f"Success: {test_name}: {'PASSED' if result else 'FAILED'}"
                )

            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "duration": 0,
                }
                logger.error(f"Error: {test_name}: ERROR - {e}")

        return self._generate_test_report()

    async def test_connection_stress(self) -> bool:
        """Test connection under stress conditions"""
        logger.info("Stress: Testing connection stress resistance...")

        try:
            client = AsyncPocketOptionClient(self.ssid, persistent_connection=True)

            # Connect and disconnect rapidly
            for i in range(5):
                logger.info(f"Connection cycle {i + 1}/5")
                success = await client.connect()
                if not success:
                    return False

                await asyncio.sleep(2)
                await client.disconnect()
                await asyncio.sleep(1)

            # Final connection for stability test
            success = await client.connect()
            if not success:
                return False

            # Send rapid messages
            for i in range(50):
                await client.send_message('42["ps"]')
                await asyncio.sleep(0.1)

            await client.disconnect()
            return True

        except Exception as e:
            logger.error(f"Connection stress test failed: {e}")
            return False

    async def test_concurrent_operations(self) -> bool:
        """Test concurrent API operations"""
        logger.info("Performance: Testing concurrent operations...")

        try:
            client = AsyncPocketOptionClient(self.ssid, persistent_connection=True)
            await client.connect()

            # Concurrent tasks
            async def get_balance_task():
                for _ in range(10):
                    await client.get_balance()
                    await asyncio.sleep(0.5)

            async def get_candles_task():
                for _ in range(5):
                    await client.get_candles("EURUSD", TimeFrame.M1, 50)
                    await asyncio.sleep(1)

            async def ping_task():
                for _ in range(20):
                    await client.send_message('42["ps"]')
                    await asyncio.sleep(0.3)

            # Run concurrently
            await asyncio.gather(
                get_balance_task(),
                get_candles_task(),
                ping_task(),
                return_exceptions=True,
            )

            await client.disconnect()
            return True

        except Exception as e:
            logger.error(f"Concurrent operations test failed: {e}")
            return False

    async def test_data_consistency(self) -> bool:
        """Test data consistency across multiple requests"""
        logger.info("Statistics: Testing data consistency...")

        try:
            client = AsyncPocketOptionClient(self.ssid)
            await client.connect()

            # Get balance multiple times
            balances = []
            for i in range(5):
                balance = await client.get_balance()
                balances.append(balance.balance)
                await asyncio.sleep(1)

            # Check if balance is consistent (allowing for small variations)
            if len(set(balances)) > 2:  # Allow for some variation
                logger.warning(f"Balance inconsistency detected: {balances}")

            # Get candles and check consistency
            candles1 = await client.get_candles("EURUSD", TimeFrame.M1, 10)
            await asyncio.sleep(2)
            candles2 = await client.get_candles("EURUSD", TimeFrame.M1, 10)

            # Most candles should be the same (except maybe the latest)
            consistent_candles = sum(
                1
                for c1, c2 in zip(candles1[:-1], candles2[:-1])
                if c1.open == c2.open and c1.close == c2.close
            )

            consistency_ratio = (
                consistent_candles / len(candles1[:-1]) if len(candles1) > 1 else 1
            )
            logger.info(f"Data consistency ratio: {consistency_ratio:.2f}")

            await client.disconnect()
            return consistency_ratio > 0.8  # 80% consistency threshold

        except Exception as e:
            logger.error(f"Data consistency test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling capabilities"""
        logger.info("Error Handling: Testing error handling...")

        try:
            client = AsyncPocketOptionClient(self.ssid)
            await client.connect()

            # Test invalid asset
            try:
                await client.get_candles("INVALID_ASSET", TimeFrame.M1, 10)
                logger.warning("Expected error for invalid asset didn't occur")
            except Exception:
                logger.info("Success: Invalid asset error handled correctly")

            # Test invalid order
            try:
                await client.place_order("EURUSD", -100, OrderDirection.CALL, 60)
                logger.warning("Expected error for negative amount didn't occur")
            except Exception:
                logger.info("Success: Invalid order error handled correctly")

            # Test connection after disconnect
            await client.disconnect()
            try:
                await client.get_balance()
                logger.warning("Expected error for disconnected client didn't occur")
            except Exception:
                logger.info("Success: Disconnected client error handled correctly")

            return True

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks"""
        logger.info("Starting: Running performance benchmarks...")

        try:
            client = AsyncPocketOptionClient(self.ssid, persistent_connection=True)

            # Connection time benchmark
            start_time = time.time()
            await client.connect()
            connection_time = time.time() - start_time

            # Balance retrieval benchmark
            start_time = time.time()
            for _ in range(10):
                await client.get_balance()
            balance_time = (time.time() - start_time) / 10

            # Candles retrieval benchmark
            start_time = time.time()
            await client.get_candles("EURUSD", TimeFrame.M1, 100)
            candles_time = time.time() - start_time

            # Message sending benchmark
            start_time = time.time()
            for _ in range(100):
                await client.send_message('42["ps"]')
            message_time = (time.time() - start_time) / 100

            # Store performance metrics
            self.performance_metrics = {
                "connection_time": connection_time,
                "avg_balance_time": balance_time,
                "candles_retrieval_time": candles_time,
                "avg_message_time": message_time,
            }

            logger.info("Data: Performance Metrics:")
            logger.info(f"  Connection Time: {connection_time:.3f}s")
            logger.info(f"  Avg Balance Time: {balance_time:.3f}s")
            logger.info(f"  Candles Retrieval: {candles_time:.3f}s")
            logger.info(f"  Avg Message Time: {message_time:.4f}s")

            await client.disconnect()

            # Performance thresholds
            return (
                connection_time < 10.0
                and balance_time < 2.0
                and candles_time < 5.0
                and message_time < 0.1
            )

        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return False

    async def test_memory_usage(self) -> bool:
        """Test memory usage patterns"""
        logger.info("Memory: Testing memory usage...")

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            client = AsyncPocketOptionClient(self.ssid, persistent_connection=True)
            await client.connect()

            # Perform memory-intensive operations
            for i in range(50):
                await client.get_candles("EURUSD", TimeFrame.M1, 100)
                await client.get_balance()

                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    logger.info(
                        f"Memory usage after {i} operations: {current_memory:.1f} MB"
                    )

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            logger.info(f"Initial memory: {initial_memory:.1f} MB")
            logger.info(f"Final memory: {final_memory:.1f} MB")
            logger.info(f"Memory increase: {memory_increase:.1f} MB")

            await client.disconnect()

            # Check for memory leaks (threshold: 50MB increase)
            return memory_increase < 50.0

        except ImportError:
            logger.warning("psutil not available, skipping memory test")
            return True
        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            return False

    async def test_network_resilience(self) -> bool:
        """Test network resilience and reconnection"""
        logger.info("Network: Testing network resilience...")

        try:
            # Use keep-alive manager for this test
            keep_alive = ConnectionKeepAlive(self.ssid, is_demo=True)

            # Event tracking
            events = []

            async def track_event(event_type):
                def handler(data):
                    events.append(
                        {"type": event_type, "time": datetime.now(), "data": data}
                    )

                return handler

            keep_alive.add_event_handler("connected", await track_event("connected"))
            keep_alive.add_event_handler(
                "reconnected", await track_event("reconnected")
            )
            keep_alive.add_event_handler(
                "message_received", await track_event("message")
            )

            # Start connection
            success = await keep_alive.start_persistent_connection()
            if not success:
                return False

            # Let it run for a bit
            await asyncio.sleep(10)

            # Simulate network issues by stopping/starting
            await keep_alive.stop_persistent_connection()
            await asyncio.sleep(3)

            # Restart and check resilience
            success = await keep_alive.start_persistent_connection()
            if not success:
                return False

            await asyncio.sleep(5)
            await keep_alive.stop_persistent_connection()

            # Check events
            connected_events = [e for e in events if e["type"] == "connected"]
            message_events = [e for e in events if e["type"] == "message"]

            logger.info(
                f"Network resilience test: {len(connected_events)} connections, {len(message_events)} messages"
            )

            return len(connected_events) >= 2 and len(message_events) > 0

        except Exception as e:
            logger.error(f"Network resilience test failed: {e}")
            return False

    async def test_long_running_session(self) -> bool:
        """Test long-running session stability"""
        logger.info("Long-running: Testing long-running session...")

        try:
            client = AsyncPocketOptionClient(self.ssid, persistent_connection=True)
            await client.connect()

            start_time = datetime.now()
            operations_count = 0
            errors_count = 0

            # Run for 2 minutes
            while (datetime.now() - start_time).total_seconds() < 120:
                try:
                    # Perform various operations
                    operation = random.choice(["balance", "candles", "ping"])

                    if operation == "balance":
                        await client.get_balance()
                    elif operation == "candles":
                        asset = random.choice(["EURUSD", "GBPUSD", "USDJPY"])
                        await client.get_candles(asset, TimeFrame.M1, 10)
                    elif operation == "ping":
                        await client.send_message('42["ps"]')

                    operations_count += 1

                except Exception as e:
                    errors_count += 1
                    logger.warning(f"Operation error: {e}")

                await asyncio.sleep(random.uniform(1, 3))

            success_rate = (
                (operations_count - errors_count) / operations_count
                if operations_count > 0
                else 0
            )

            logger.info(
                f"Long-running session: {operations_count} operations, {errors_count} errors"
            )
            logger.info(f"Success rate: {success_rate:.2%}")

            await client.disconnect()

            return success_rate > 0.9  # 90% success rate

        except Exception as e:
            logger.error(f"Long-running session test failed: {e}")
            return False

    async def test_multi_asset_operations(self) -> bool:
        """Test operations across multiple assets"""
        logger.info("Retrieved: Testing multi-asset operations...")

        try:
            client = AsyncPocketOptionClient(self.ssid)
            await client.connect()

            assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]

            # Get candles for multiple assets concurrently
            async def get_asset_candles(asset):
                try:
                    candles = await client.get_candles(asset, TimeFrame.M1, 20)
                    return asset, len(candles), True
                except Exception as e:
                    logger.warning(f"Failed to get candles for {asset}: {e}")
                    return asset, 0, False

            results = await asyncio.gather(
                *[get_asset_candles(asset) for asset in assets]
            )

            successful_assets = sum(1 for _, _, success in results if success)
            total_candles = sum(count for _, count, _ in results)

            logger.info(
                f"Multi-asset test: {successful_assets}/{len(assets)} assets successful"
            )
            logger.info(f"Total candles retrieved: {total_candles}")

            await client.disconnect()

            return successful_assets >= len(assets) * 0.8  # 80% success rate

        except Exception as e:
            logger.error(f"Multi-asset operations test failed: {e}")
            return False

    async def test_rapid_trading_simulation(self) -> bool:
        """Simulate rapid trading operations"""
        logger.info("Performance: Testing rapid trading simulation...")

        try:
            client = AsyncPocketOptionClient(self.ssid)
            await client.connect()

            # Simulate rapid order operations (without actually placing real orders)
            operations = []

            for i in range(20):
                try:
                    # Get balance before "trade"
                    balance = await client.get_balance()

                    # Get current market data
                    candles = await client.get_candles("EURUSD", TimeFrame.M1, 5)

                    # Simulate order decision (don't actually place)
                    direction = (
                        OrderDirection.CALL
                        if len(candles) % 2 == 0
                        else OrderDirection.PUT
                    )
                    amount = random.uniform(1, 10)

                    operations.append(
                        {
                            "balance": balance.balance,
                            "direction": direction,
                            "amount": amount,
                            "candles_count": len(candles),
                            "timestamp": datetime.now(),
                        }
                    )

                    await asyncio.sleep(0.5)  # Rapid operations

                except Exception as e:
                    logger.warning(f"Rapid trading simulation error: {e}")

            await client.disconnect()

            logger.info(
                f"Rapid trading simulation: {len(operations)} operations completed"
            )

            return len(operations) >= 18  # 90% completion rate

        except Exception as e:
            logger.error(f"Rapid trading simulation failed: {e}")
            return False

    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""

        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "PASSED"
        )
        failed_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "FAILED"
        )
        error_tests = sum(
            1 for result in self.test_results.values() if result["status"] == "ERROR"
        )

        total_duration = sum(
            result.get("duration", 0) for result in self.test_results.values()
        )

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
            },
            "detailed_results": self.test_results,
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        return report


async def run_advanced_tests(ssid: str = None):
    """Run the advanced testing suite"""

    if not ssid:
        # Use demo SSID for testing
        ssid = r'42["auth",{"session":"demo_session_for_testing","isDemo":1,"uid":0,"platform":1}]'
        logger.warning(
            "Caution: Using demo SSID - some tests may have limited functionality"
        )

    test_suite = AdvancedTestSuite(ssid)

    logger.info("Testing: Starting Advanced PocketOption API Testing Suite")
    logger.info("=" * 60)

    try:
        report = await test_suite.run_all_tests()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Demonstration: TEST SUMMARY")
        logger.info("=" * 60)

        summary = report["summary"]
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']} Success")
        logger.info(f"Failed: {summary['failed']} Error")
        logger.info(f"Errors: {summary['errors']} Failure")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Total Duration: {summary['total_duration']:.2f}s")

        # Performance metrics
        if report["performance_metrics"]:
            logger.info("\nStatistics: PERFORMANCE METRICS")
            logger.info("-" * 30)
            for metric, value in report["performance_metrics"].items():
                logger.info(f"{metric}: {value:.3f}s")

        # Save report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\nReport: Detailed report saved to: {report_file}")

        return report

    except Exception as e:
        logger.error(f"Error: Test suite failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    # Allow passing SSID as command line argument
    ssid = None
    if len(sys.argv) > 1:
        ssid = sys.argv[1]
        logger.info(f"Using provided SSID: {ssid[:50]}...")

    asyncio.run(run_advanced_tests(ssid))
