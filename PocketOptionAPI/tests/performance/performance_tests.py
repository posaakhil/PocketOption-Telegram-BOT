"""
Performance Tests for PocketOption Async API
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from loguru import logger

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection


class PerformanceTester:
    """Performance testing utilities for the async API"""

    def __init__(self, session_id: str, is_demo: bool = True):
        self.session_id = session_id
        self.is_demo = is_demo
        self.results: Dict[str, List[float]] = {}

    async def test_connection_performance(
        self, iterations: int = 5
    ) -> Dict[str, float]:
        """Test connection establishment performance"""
        logger.info(f"Testing connection performance ({iterations} iterations)")

        connection_times = []

        for i in range(iterations):
            start_time = time.time()

            client = AsyncPocketOptionClient(
                session_id=self.session_id, is_demo=self.is_demo
            )

            try:
                await client.connect()
                if client.is_connected:
                    connection_time = time.time() - start_time
                    connection_times.append(connection_time)
                    logger.success(f"Connection {i + 1}: {connection_time:.3f}s")
                else:
                    logger.warning(f"Connection {i + 1}: Failed")

            except Exception as e:
                logger.error(f"Connection {i + 1}: Error - {e}")
            finally:
                await client.disconnect()
                await asyncio.sleep(1)  # Cool down

        if connection_times:
            return {
                "avg_time": statistics.mean(connection_times),
                "min_time": min(connection_times),
                "max_time": max(connection_times),
                "std_dev": statistics.stdev(connection_times)
                if len(connection_times) > 1
                else 0,
                "success_rate": len(connection_times) / iterations * 100,
            }
        else:
            return {"success_rate": 0}

    async def test_order_placement_performance(
        self, iterations: int = 10
    ) -> Dict[str, float]:
        """Test order placement performance"""
        logger.info(f"Testing order placement performance ({iterations} iterations)")

        client = AsyncPocketOptionClient(
            session_id=self.session_id, is_demo=self.is_demo
        )

        order_times = []
        successful_orders = 0

        try:
            await client.connect()

            if not client.is_connected:
                logger.error("Failed to connect for order testing")
                return {"success_rate": 0}

            # Wait for balance
            await asyncio.sleep(2)

            for i in range(iterations):
                start_time = time.time()

                try:
                    order = await client.place_order(
                        asset="EURUSD_otc",
                        amount=1.0,
                        direction=OrderDirection.CALL,
                        duration=60,
                    )

                    if order:
                        order_time = time.time() - start_time
                        order_times.append(order_time)
                        successful_orders += 1
                        logger.success(f"Order {i + 1}: {order_time:.3f}s")
                    else:
                        logger.warning(f"Order {i + 1}: Failed (no response)")

                except Exception as e:
                    logger.error(f"Order {i + 1}: Error - {e}")

                await asyncio.sleep(0.1)  # Small delay between orders

        finally:
            await client.disconnect()

        if order_times:
            return {
                "avg_time": statistics.mean(order_times),
                "min_time": min(order_times),
                "max_time": max(order_times),
                "std_dev": statistics.stdev(order_times) if len(order_times) > 1 else 0,
                "success_rate": successful_orders / iterations * 100,
                "orders_per_second": 1 / statistics.mean(order_times)
                if order_times
                else 0,
            }
        else:
            return {"success_rate": 0}

    async def test_data_retrieval_performance(self) -> Dict[str, float]:
        """Test data retrieval performance"""
        logger.info("Testing data retrieval performance")

        client = AsyncPocketOptionClient(
            session_id=self.session_id, is_demo=self.is_demo
        )

        operations = {
            "balance": lambda: client.get_balance(),
            "candles": lambda: client.get_candles("EURUSD_otc", 60, 100),
            "active_orders": lambda: client.get_active_orders(),
        }

        results = {}

        try:
            await client.connect()

            if not client.is_connected:
                logger.error("Failed to connect for data testing")
                return {}

            await asyncio.sleep(2)  # Wait for initialization

            for operation_name, operation in operations.items():
                times = []

                for i in range(5):  # 5 iterations per operation
                    start_time = time.time()

                    try:
                        await operation()
                        operation_time = time.time() - start_time
                        times.append(operation_time)
                        logger.success(
                            f"{operation_name} {i + 1}: {operation_time:.3f}s"
                        )

                    except Exception as e:
                        logger.error(f"{operation_name} {i + 1}: Error - {e}")

                    await asyncio.sleep(0.1)

                if times:
                    results[operation_name] = {
                        "avg_time": statistics.mean(times),
                        "min_time": min(times),
                        "max_time": max(times),
                    }

        finally:
            await client.disconnect()

        return results

    async def test_concurrent_operations(
        self, concurrency_level: int = 5
    ) -> Dict[str, Any]:
        """Test concurrent operations performance"""
        logger.info(f"Testing concurrent operations (level: {concurrency_level})")

        async def perform_operation(operation_id: int):
            client = AsyncPocketOptionClient(
                session_id=self.session_id, is_demo=self.is_demo
            )

            start_time = time.time()

            try:
                await client.connect()

                if client.is_connected:
                    balance = await client.get_balance()
                    operation_time = time.time() - start_time
                    return {
                        "operation_id": operation_id,
                        "success": True,
                        "time": operation_time,
                        "balance": balance.balance if balance else None,
                    }
                else:
                    return {
                        "operation_id": operation_id,
                        "success": False,
                        "time": time.time() - start_time,
                        "error": "Connection failed",
                    }

            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "time": time.time() - start_time,
                    "error": str(e),
                }
            finally:
                await client.disconnect()

        # Run concurrent operations
        start_time = time.time()
        tasks = [perform_operation(i) for i in range(concurrency_level)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze results
        successful_operations = [
            r for r in results if isinstance(r, dict) and r.get("success")
        ]
        failed_operations = [
            r for r in results if not (isinstance(r, dict) and r.get("success"))
        ]

        if successful_operations:
            operation_times = [r["time"] for r in successful_operations]

            return {
                "total_time": total_time,
                "success_rate": len(successful_operations) / concurrency_level * 100,
                "avg_operation_time": statistics.mean(operation_times),
                "min_operation_time": min(operation_times),
                "max_operation_time": max(operation_times),
                "operations_per_second": len(successful_operations) / total_time,
                "failed_count": len(failed_operations),
            }
        else:
            return {
                "total_time": total_time,
                "success_rate": 0,
                "failed_count": len(failed_operations),
            }

    async def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        logger.info("Starting comprehensive performance tests...")

        report = []
        report.append("=" * 60)
        report.append("POCKETOPTION ASYNC API PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        # Test 1: Connection Performance
        report.append("Connection: CONNECTION PERFORMANCE")
        report.append("-" * 30)
        try:
            conn_results = await self.test_connection_performance()
            if conn_results.get("success_rate", 0) > 0:
                report.append(
                    f"Success: Average Connection Time: {conn_results['avg_time']:.3f}s"
                )
                report.append(
                    f"Success: Min Connection Time: {conn_results['min_time']:.3f}s"
                )
                report.append(
                    f"Success: Max Connection Time: {conn_results['max_time']:.3f}s"
                )
                report.append(
                    f"Success: Success Rate: {conn_results['success_rate']:.1f}%"
                )
                report.append(
                    f"Success: Standard Deviation: {conn_results['std_dev']:.3f}s"
                )
            else:
                report.append("Error: Connection tests failed")
        except Exception as e:
            report.append(f"Error: Connection test error: {e}")

        report.append("")

        # Test 2: Data Retrieval Performance
        report.append("Statistics: DATA RETRIEVAL PERFORMANCE")
        report.append("-" * 35)
        try:
            data_results = await self.test_data_retrieval_performance()
            for operation, stats in data_results.items():
                report.append(f"  {operation.upper()}:")
                report.append(f"    Average: {stats['avg_time']:.3f}s")
                report.append(
                    f"    Range: {stats['min_time']:.3f}s - {stats['max_time']:.3f}s"
                )
        except Exception as e:
            report.append(f"Error: Data retrieval test error: {e}")

        report.append("")

        # Test 3: Concurrent Operations
        report.append("Performance: CONCURRENT OPERATIONS")
        report.append("-" * 25)
        try:
            concurrent_results = await self.test_concurrent_operations()
            if concurrent_results.get("success_rate", 0) > 0:
                report.append(
                    f"Success: Success Rate: {concurrent_results['success_rate']:.1f}%"
                )
                report.append(
                    f"Success: Operations/Second: {concurrent_results['operations_per_second']:.2f}"
                )
                report.append(
                    f"Success: Avg Operation Time: {concurrent_results['avg_operation_time']:.3f}s"
                )
                report.append(
                    f"Success: Total Time: {concurrent_results['total_time']:.3f}s"
                )
            else:
                report.append("Error: Concurrent operations failed")
        except Exception as e:
            report.append(f"Error: Concurrent test error: {e}")

        report.append("")
        report.append("=" * 60)
        report.append(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        return "\n".join(report)


async def main():
    """Run performance tests"""
    # Use test session (replace with real session for full tests)
    tester = PerformanceTester(
        session_id="n1p5ah5u8t9438rbunpgrq0hlq",  # Replace with your session ID
        is_demo=True,
    )

    report = await tester.generate_performance_report()
    print(report)

    # Save report to file
    with open("performance_report.txt", "w") as f:
        f.write(report)

    logger.success("Performance report saved to performance_report.txt")


if __name__ == "__main__":
    asyncio.run(main())
