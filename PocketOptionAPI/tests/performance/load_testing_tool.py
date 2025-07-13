"""
Load Testing and Stress Testing Tool for PocketOption Async API
"""

import asyncio
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
from loguru import logger

from pocketoptionapi_async.client import AsyncPocketOptionClient
from pocketoptionapi_async.models import OrderDirection, TimeFrame
from pocketoptionapi_async.connection_keep_alive import ConnectionKeepAlive


@dataclass
class LoadTestResult:
    """Result of a load test operation"""

    operation_type: str
    start_time: datetime
    end_time: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Any] = None


@dataclass
class LoadTestConfig:
    """Load test configuration"""

    concurrent_clients: int = 5
    operations_per_client: int = 20
    operation_delay: float = 1.0
    test_duration_minutes: int = 5
    use_persistent_connection: bool = True
    include_trading_operations: bool = False
    stress_mode: bool = False


class LoadTester:
    """Advanced load testing framework"""

    def __init__(self, ssid: str, is_demo: bool = True):
        self.ssid = ssid
        self.is_demo = is_demo

        # Test state
        self.test_results: List[LoadTestResult] = []
        self.active_clients: List[AsyncPocketOptionClient] = []
        self.test_start_time: Optional[datetime] = None
        self.test_end_time: Optional[datetime] = None

        # Statistics
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)

        # Real-time monitoring
        self.operations_per_second: deque = deque(maxlen=60)  # Last 60 seconds
        self.current_operations = 0
        self.peak_operations_per_second = 0

    async def run_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run comprehensive load test"""
        logger.info("Starting Load Test")
        logger.info(
            f"Config: {config.concurrent_clients} clients, {config.operations_per_client} ops/client"
        )

        self.test_start_time = datetime.now()

        try:
            if config.stress_mode:
                return await self._run_stress_test(config)
            else:
                return await self._run_standard_load_test(config)

        finally:
            self.test_end_time = datetime.now()
            await self._cleanup_clients()

    async def _run_standard_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run standard load test with concurrent clients"""

        # Create client tasks
        client_tasks = []
        for i in range(config.concurrent_clients):
            task = asyncio.create_task(
                self._run_client_operations(
                    client_id=i,
                    operations_count=config.operations_per_client,
                    delay=config.operation_delay,
                    persistent=config.use_persistent_connection,
                    include_trading=config.include_trading_operations,
                )
            )
            client_tasks.append(task)

        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_operations())

        # Run all client tasks
        logger.info(f"Persistent: Running {len(client_tasks)} concurrent clients...")

        try:
            await asyncio.gather(*client_tasks, return_exceptions=True)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        return self._generate_load_test_report()

    async def _run_stress_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run stress test with extreme conditions"""
        logger.info("Error: Running STRESS TEST mode!")

        # Stress test phases
        phases = [
            ("Ramp Up", config.concurrent_clients // 3, 0.1),
            ("Peak Load", config.concurrent_clients, 0.05),
            ("Extreme Load", config.concurrent_clients * 2, 0.01),
            ("Cool Down", config.concurrent_clients // 2, 0.5),
        ]

        for phase_name, clients, delay in phases:
            logger.info(
                f"Stress: Stress Phase: {phase_name} ({clients} clients, {delay}s delay)"
            )

            # Create tasks for this phase
            phase_tasks = []
            for i in range(clients):
                task = asyncio.create_task(
                    self._run_stress_client(
                        client_id=f"{phase_name}_{i}",
                        operations_count=config.operations_per_client // 2,
                        delay=delay,
                        persistent=config.use_persistent_connection,
                    )
                )
                phase_tasks.append(task)

            # Run phase
            try:
                await asyncio.wait_for(
                    asyncio.gather(*phase_tasks, return_exceptions=True),
                    timeout=60,  # 1 minute per phase
                )
            except asyncio.TimeoutError:
                logger.warning(f"Long-running: Stress phase {phase_name} timed out")
                # Cancel remaining tasks
                for task in phase_tasks:
                    if not task.done():
                        task.cancel()

            # Brief pause between phases
            await asyncio.sleep(5)

        return self._generate_load_test_report()

    async def _run_client_operations(
        self,
        client_id: int,
        operations_count: int,
        delay: float,
        persistent: bool,
        include_trading: bool,
    ) -> None:
        """Run operations for a single client"""
        client = None

        try:
            # Create and connect client
            client = AsyncPocketOptionClient(
                self.ssid,
                is_demo=self.is_demo,
                persistent_connection=persistent,
                auto_reconnect=True,
            )

            self.active_clients.append(client)

            # Connect
            connect_start = datetime.now()
            success = await client.connect()
            connect_end = datetime.now()

            if success:
                self._record_result(
                    LoadTestResult(
                        operation_type="connect",
                        start_time=connect_start,
                        end_time=connect_end,
                        duration=(connect_end - connect_start).total_seconds(),
                        success=True,
                        response_data={"client_id": client_id},
                    )
                )

                logger.info(f"Success: Client {client_id} connected")
            else:
                self._record_result(
                    LoadTestResult(
                        operation_type="connect",
                        start_time=connect_start,
                        end_time=connect_end,
                        duration=(connect_end - connect_start).total_seconds(),
                        success=False,
                        error_message="Connection failed",
                    )
                )
                return

            # Run operations
            for op_num in range(operations_count):
                try:
                    # Choose operation type
                    operation_type = self._choose_operation_type(include_trading)

                    # Execute operation
                    await self._execute_operation(client, client_id, operation_type)

                    # Delay between operations
                    if delay > 0:
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(
                        f"Error: Client {client_id} operation {op_num} failed: {e}"
                    )
                    self._record_result(
                        LoadTestResult(
                            operation_type="unknown",
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            duration=0,
                            success=False,
                            error_message=str(e),
                        )
                    )

        except Exception as e:
            logger.error(f"Error: Client {client_id} failed: {e}")

        finally:
            if client:
                try:
                    await client.disconnect()
                except:
                    pass

    async def _run_stress_client(
        self, client_id: str, operations_count: int, delay: float, persistent: bool
    ) -> None:
        """Run stress operations for a single client"""

        # Create keep-alive manager for stress testing
        keep_alive = None

        try:
            keep_alive = ConnectionKeepAlive(self.ssid, is_demo=self.is_demo)

            # Connect
            connect_start = datetime.now()
            success = await keep_alive.start_persistent_connection()
            connect_end = datetime.now()

            if not success:
                self._record_result(
                    LoadTestResult(
                        operation_type="stress_connect",
                        start_time=connect_start,
                        end_time=connect_end,
                        duration=(connect_end - connect_start).total_seconds(),
                        success=False,
                        error_message="Stress connection failed",
                    )
                )
                return

            # Rapid-fire operations
            for op_num in range(operations_count):
                try:
                    op_start = datetime.now()

                    # Send multiple messages rapidly
                    for _ in range(3):
                        await keep_alive.send_message('42["ps"]')
                        await asyncio.sleep(0.01)  # 10ms between messages

                    op_end = datetime.now()

                    self._record_result(
                        LoadTestResult(
                            operation_type="stress_rapid_ping",
                            start_time=op_start,
                            end_time=op_end,
                            duration=(op_end - op_start).total_seconds(),
                            success=True,
                            response_data={"client_id": client_id, "messages": 3},
                        )
                    )

                    if delay > 0:
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(
                        f"Error: Stress client {client_id} operation {op_num} failed: {e}"
                    )

        except Exception as e:
            logger.error(f"Error: Stress client {client_id} failed: {e}")

        finally:
            if keep_alive:
                try:
                    await keep_alive.stop_persistent_connection()
                except:
                    pass

    def _choose_operation_type(self, include_trading: bool) -> str:
        """Choose random operation type"""
        basic_operations = ["balance", "candles", "ping", "market_data"]

        if include_trading:
            trading_operations = ["place_order", "check_order", "get_orders"]
            basic_operations.extend(trading_operations)

        return random.choice(basic_operations)

    async def _execute_operation(
        self, client: AsyncPocketOptionClient, client_id: int, operation_type: str
    ) -> None:
        """Execute a specific operation and record results"""
        start_time = datetime.now()

        try:
            if operation_type == "balance":
                balance = await client.get_balance()
                result_data = {"balance": balance.balance if balance else None}

            elif operation_type == "candles":
                asset = random.choice(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
                timeframe = random.choice([TimeFrame.M1, TimeFrame.M5, TimeFrame.M15])
                candles = await client.get_candles(
                    asset, timeframe, random.randint(10, 50)
                )
                result_data = {"asset": asset, "candles_count": len(candles)}

            elif operation_type == "ping":
                await client.send_message('42["ps"]')
                result_data = {"message": "ping"}

            elif operation_type == "market_data":
                # Simulate getting multiple market data
                for asset in ["EURUSD", "GBPUSD"]:
                    await client.get_candles(asset, TimeFrame.M1, 5)
                result_data = {"assets": 2}

            elif operation_type == "place_order":
                # Simulate order (don't actually place in demo)
                asset = random.choice(["EURUSD", "GBPUSD"])
                amount = random.uniform(1, 10)
                direction = random.choice([OrderDirection.CALL, OrderDirection.PUT])
                result_data = {
                    "asset": asset,
                    "amount": amount,
                    "direction": direction.value,
                }

            elif operation_type == "check_order":
                # Simulate order check
                await asyncio.sleep(0.1)  # Simulate API call
                result_data = {"orders": 0}

            elif operation_type == "get_orders":
                # Simulate getting orders
                await asyncio.sleep(0.1)  # Simulate API call
                result_data = {"active_orders": 0}

            else:
                result_data = {}

            end_time = datetime.now()

            self._record_result(
                LoadTestResult(
                    operation_type=operation_type,
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    success=True,
                    response_data=result_data,
                )
            )

            self.current_operations += 1

        except Exception as e:
            end_time = datetime.now()

            self._record_result(
                LoadTestResult(
                    operation_type=operation_type,
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time).total_seconds(),
                    success=False,
                    error_message=str(e),
                )
            )

    async def _monitor_operations(self):
        """Monitor operations per second"""
        while True:
            try:
                await asyncio.sleep(1)

                # Record operations per second
                ops_this_second = self.current_operations
                self.operations_per_second.append(ops_this_second)

                # Update peak
                if ops_this_second > self.peak_operations_per_second:
                    self.peak_operations_per_second = ops_this_second

                # Reset counter
                self.current_operations = 0

                # Log every 10 seconds
                if len(self.operations_per_second) % 10 == 0:
                    avg_ops = statistics.mean(list(self.operations_per_second)[-10:])
                    logger.info(
                        f"Statistics: Avg ops/sec (last 10s): {avg_ops:.1f}, Peak: {self.peak_operations_per_second}"
                    )

            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _record_result(self, result: LoadTestResult):
        """Record test result"""
        self.test_results.append(result)

        # Update statistics
        if result.success:
            self.success_counts[result.operation_type] += 1
            self.operation_stats[result.operation_type].append(result.duration)
        else:
            self.error_counts[result.operation_type] += 1

    async def _cleanup_clients(self):
        """Clean up all active clients"""
        logger.info("Cleaning up clients...")

        cleanup_tasks = []
        for client in self.active_clients:
            if client.is_connected:
                cleanup_tasks.append(asyncio.create_task(client.disconnect()))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.active_clients.clear()
        logger.info("Success: Cleanup completed")

    def _generate_load_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report"""

        if not self.test_start_time or not self.test_end_time:
            return {"error": "Test timing not available"}

        total_duration = (self.test_end_time - self.test_start_time).total_seconds()
        total_operations = len(self.test_results)
        successful_operations = sum(1 for r in self.test_results if r.success)
        failed_operations = total_operations - successful_operations

        # Calculate operation statistics
        operation_analysis = {}
        for op_type, durations in self.operation_stats.items():
            if durations:
                operation_analysis[op_type] = {
                    "count": len(durations),
                    "success_count": self.success_counts[op_type],
                    "error_count": self.error_counts[op_type],
                    "success_rate": self.success_counts[op_type]
                    / (self.success_counts[op_type] + self.error_counts[op_type]),
                    "avg_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "median_duration": statistics.median(durations),
                    "p95_duration": sorted(durations)[int(len(durations) * 0.95)]
                    if len(durations) > 20
                    else max(durations),
                }

        # Performance metrics
        avg_ops_per_second = (
            total_operations / total_duration if total_duration > 0 else 0
        )

        # Error analysis
        error_summary = {}
        for result in self.test_results:
            if not result.success and result.error_message:
                error_type = (
                    result.error_message.split(":")[0]
                    if ":" in result.error_message
                    else result.error_message
                )
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

        report = {
            "test_summary": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": self.test_end_time.isoformat(),
                "total_duration": total_duration,
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / total_operations
                if total_operations > 0
                else 0,
                "avg_operations_per_second": avg_ops_per_second,
                "peak_operations_per_second": self.peak_operations_per_second,
            },
            "operation_analysis": operation_analysis,
            "error_summary": error_summary,
            "performance_metrics": {
                "operations_per_second_history": list(self.operations_per_second),
                "peak_throughput": self.peak_operations_per_second,
                "avg_throughput": avg_ops_per_second,
            },
            "recommendations": self._generate_recommendations(
                operation_analysis,
                avg_ops_per_second,
                successful_operations / total_operations if total_operations > 0 else 0,
            ),
        }

        return report

    def _generate_recommendations(
        self, operation_analysis: Dict, avg_throughput: float, success_rate: float
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        if success_rate < 0.95:
            recommendations.append(
                f"Low success rate ({success_rate:.1%}). Check network stability and API limits."
            )

        if avg_throughput < 1:
            recommendations.append(
                "Low throughput detected. Consider using persistent connections."
            )

        # Check slow operations
        slow_operations = []
        for op_type, stats in operation_analysis.items():
            if stats["avg_duration"] > 2.0:
                slow_operations.append(f"{op_type} ({stats['avg_duration']:.2f}s avg)")

        if slow_operations:
            recommendations.append(
                f"Slow operations detected: {', '.join(slow_operations)}"
            )

        # Check high error rate operations
        error_operations = []
        for op_type, stats in operation_analysis.items():
            if stats["success_rate"] < 0.9:
                error_operations.append(
                    f"{op_type} ({stats['success_rate']:.1%} success)"
                )

        if error_operations:
            recommendations.append(
                f"High error rate operations: {', '.join(error_operations)}"
            )

        if not recommendations:
            recommendations.append(
                "System performance is good. No major issues detected."
            )

        return recommendations


async def run_load_test_demo(ssid: str = None):
    """Run load testing demonstration"""

    if not ssid:
        ssid = r'42["auth",{"session":"demo_session_for_load_test","isDemo":1,"uid":0,"platform":1}]'
        logger.warning("Caution: Using demo SSID for load testing")

    logger.info("Starting Load Testing Demo")

    # Create load tester
    load_tester = LoadTester(ssid, is_demo=True)

    # Test configurations
    test_configs = [
        LoadTestConfig(
            concurrent_clients=3,
            operations_per_client=10,
            operation_delay=0.5,
            use_persistent_connection=False,
            stress_mode=False,
        ),
        LoadTestConfig(
            concurrent_clients=5,
            operations_per_client=15,
            operation_delay=0.2,
            use_persistent_connection=True,
            stress_mode=False,
        ),
        LoadTestConfig(
            concurrent_clients=2,
            operations_per_client=5,
            operation_delay=0.1,
            use_persistent_connection=True,
            stress_mode=True,
        ),
    ]

    all_reports = []

    for i, config in enumerate(test_configs, 1):
        logger.info(f"\nTesting: Running Load Test {i}/{len(test_configs)}")
        logger.info(f"Configuration: {config}")

        try:
            report = await load_tester.run_load_test(config)
            all_reports.append(report)

            # Print summary
            summary = report["test_summary"]
            logger.info(f"Success: Test {i} completed:")
            logger.info(f"  Duration: {summary['total_duration']:.2f}s")
            logger.info(f"  Operations: {summary['total_operations']}")
            logger.info(f"  Success Rate: {summary['success_rate']:.1%}")
            logger.info(
                f"  Throughput: {summary['avg_operations_per_second']:.1f} ops/sec"
            )

            # Brief pause between tests
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error: Load test {i} failed: {e}")

    # Generate comparison report
    if all_reports:
        comparison_report = {
            "test_comparison": [],
            "best_performance": {},
            "overall_recommendations": [],
        }

        best_throughput = 0
        best_success_rate = 0

        for i, report in enumerate(all_reports, 1):
            summary = report["test_summary"]
            comparison_report["test_comparison"].append(
                {
                    "test_number": i,
                    "throughput": summary["avg_operations_per_second"],
                    "success_rate": summary["success_rate"],
                    "total_operations": summary["total_operations"],
                    "duration": summary["total_duration"],
                }
            )

            if summary["avg_operations_per_second"] > best_throughput:
                best_throughput = summary["avg_operations_per_second"]
                comparison_report["best_performance"]["throughput"] = f"Test {i}"

            if summary["success_rate"] > best_success_rate:
                best_success_rate = summary["success_rate"]
                comparison_report["best_performance"]["reliability"] = f"Test {i}"

        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, report in enumerate(all_reports, 1):
            report_file = f"load_test_{i}_{timestamp}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report: Test {i} report saved to: {report_file}")

        comparison_file = f"load_test_comparison_{timestamp}.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_report, f, indent=2, default=str)

        logger.info(f"Statistics: Comparison report saved to: {comparison_file}")

        # Final summary
        logger.info("\nCompleted: LOAD TESTING SUMMARY")
        logger.info("=" * 50)
        logger.info(
            f"Best Throughput: {comparison_report['best_performance'].get('throughput', 'N/A')}"
        )
        logger.info(
            f"Best Reliability: {comparison_report['best_performance'].get('reliability', 'N/A')}"
        )

        return all_reports

    return []


if __name__ == "__main__":
    import sys

    # Allow passing SSID as command line argument
    ssid = None
    if len(sys.argv) > 1:
        ssid = sys.argv[1]
        logger.info(f"Using provided SSID: {ssid[:50]}...")

    asyncio.run(run_load_test_demo(ssid))
