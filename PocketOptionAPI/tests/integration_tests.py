"""
Integration Testing Script
Tests all components of the PocketOption Async API working together
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from pocketoptionapi_async.client import AsyncPocketOptionClient
from pocketoptionapi_async.models import TimeFrame
from pocketoptionapi_async.connection_keep_alive import ConnectionKeepAlive
from pocketoptionapi_async.connection_monitor import ConnectionMonitor
from performance.load_testing_tool import LoadTester, LoadTestConfig


class IntegrationTester:
    """Comprehensive integration testing"""

    def __init__(self, ssid: str):
        self.ssid = ssid
        self.test_results = {}
        self.start_time = datetime.now()

    async def run_full_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting Full Integration Testing Suite")
        logger.info("=" * 60)

        # Test phases
        test_phases = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("SSID Format Compatibility", self.test_ssid_formats),
            ("Persistent Connection Integration", self.test_persistent_integration),
            ("Keep-Alive Functionality", self.test_keep_alive_integration),
            ("Monitoring Integration", self.test_monitoring_integration),
            ("Multi-Client Scenarios", self.test_multi_client_scenarios),
            ("Error Recovery", self.test_error_recovery),
            ("Performance Under Load", self.test_performance_integration),
            ("Data Consistency", self.test_data_consistency_integration),
            ("Long-Running Stability", self.test_long_running_stability),
        ]

        for phase_name, phase_func in test_phases:
            logger.info(f"\n🔍 {phase_name}")
            logger.info("-" * 40)

            try:
                start_time = time.time()
                result = await phase_func()
                duration = time.time() - start_time

                self.test_results[phase_name] = {
                    "status": "PASSED" if result["success"] else "FAILED",
                    "duration": duration,
                    "details": result,
                }

                status_emoji = "" if result["success"] else "❌"
                logger.info(
                    f"{status_emoji} {phase_name}: {'PASSED' if result['success'] else 'FAILED'} ({duration:.2f}s)"
                )

                if not result["success"]:
                    logger.error(f"   Error: {result.get('error', 'Unknown error')}")

            except Exception as e:
                self.test_results[phase_name] = {
                    "status": "ERROR",
                    "duration": 0,
                    "error": str(e),
                }
                logger.error(f"💥 {phase_name}: ERROR - {e}")

        return self._generate_integration_report()

    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity across all client types"""
        try:
            results = {}

            # Test regular client
            logger.info("Testing regular AsyncPocketOptionClient...")
            client = AsyncPocketOptionClient(self.ssid, is_demo=True)
            success = await client.connect()
            if success:
                balance = await client.get_balance()
                results["regular_client"] = {
                    "connected": True,
                    "balance_retrieved": balance is not None,
                }
                await client.disconnect()
            else:
                results["regular_client"] = {
                    "connected": False,
                    "balance_retrieved": False,
                }

            # Test persistent client
            logger.info("Testing persistent connection client...")
            persistent_client = AsyncPocketOptionClient(
                self.ssid, is_demo=True, persistent_connection=True
            )
            success = await persistent_client.connect()
            if success:
                balance = await persistent_client.get_balance()
                results["persistent_client"] = {
                    "connected": True,
                    "balance_retrieved": balance is not None,
                }
                await persistent_client.disconnect()
            else:
                results["persistent_client"] = {
                    "connected": False,
                    "balance_retrieved": False,
                }

            # Test keep-alive manager
            logger.info("Testing ConnectionKeepAlive...")
            keep_alive = ConnectionKeepAlive(self.ssid, is_demo=True)
            success = await keep_alive.start_persistent_connection()
            if success:
                # Test message sending
                message_sent = await keep_alive.send_message('42["ps"]')
                results["keep_alive"] = {
                    "connected": True,
                    "message_sent": message_sent,
                }
                await keep_alive.stop_persistent_connection()
            else:
                results["keep_alive"] = {"connected": False, "message_sent": False}

            # Evaluate overall success
            all_connected = all(r.get("connected", False) for r in results.values())

            return {
                "success": all_connected,
                "details": results,
                "message": f"Connectivity test: {len([r for r in results.values() if r.get('connected', False)])}/{len(results)} clients connected",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_ssid_formats(self) -> Dict[str, Any]:
        """Test different SSID format compatibility"""
        try:
            # Test different SSID formats
            ssid_formats = [
                # Complete format (what we use)
                self.ssid,
                # Alternative format test (would need valid session)
                # r'42["auth",{"session":"alternative_session","isDemo":1,"uid":123,"platform":1}]'
            ]

            results = {}

            for i, ssid_format in enumerate(ssid_formats):
                logger.info(f"Testing SSID format {i + 1}...")
                try:
                    client = AsyncPocketOptionClient(ssid_format, is_demo=True)
                    success = await client.connect()

                    if success:
                        # Test basic operation
                        balance = await client.get_balance()
                        results[f"format_{i + 1}"] = {
                            "connected": True,
                            "authenticated": balance is not None,
                            "format": ssid_format[:50] + "..."
                            if len(ssid_format) > 50
                            else ssid_format,
                        }
                        await client.disconnect()
                    else:
                        results[f"format_{i + 1}"] = {
                            "connected": False,
                            "authenticated": False,
                            "format": ssid_format[:50] + "..."
                            if len(ssid_format) > 50
                            else ssid_format,
                        }

                except Exception as e:
                    results[f"format_{i + 1}"] = {
                        "connected": False,
                        "authenticated": False,
                        "error": str(e),
                        "format": ssid_format[:50] + "..."
                        if len(ssid_format) > 50
                        else ssid_format,
                    }

            # At least one format should work
            any_success = any(r.get("connected", False) for r in results.values())

            return {
                "success": any_success,
                "details": results,
                "message": f"SSID format test: {len([r for r in results.values() if r.get('connected', False)])}/{len(results)} formats successful",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_persistent_integration(self) -> Dict[str, Any]:
        """Test persistent connection integration"""
        try:
            logger.info("Testing persistent connection features...")

            client = AsyncPocketOptionClient(
                self.ssid, is_demo=True, persistent_connection=True, auto_reconnect=True
            )

            # Connect
            success = await client.connect(persistent=True)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to establish persistent connection",
                }

            # Test multiple operations
            operations_successful = 0
            total_operations = 10

            for i in range(total_operations):
                try:
                    # Alternate between different operations
                    if i % 3 == 0:
                        balance = await client.get_balance()
                        if balance:
                            operations_successful += 1
                    elif i % 3 == 1:
                        candles = await client.get_candles("EURUSD", TimeFrame.M1, 5)
                        if len(candles) > 0:
                            operations_successful += 1
                    else:
                        success = await client.send_message('42["ps"]')
                        if success:
                            operations_successful += 1

                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Operation {i} failed: {e}")

            # Test connection stats
            stats = client.get_connection_stats()

            await client.disconnect()

            success_rate = operations_successful / total_operations

            return {
                "success": success_rate > 0.8,  # 80% success rate
                "details": {
                    "operations_successful": operations_successful,
                    "total_operations": total_operations,
                    "success_rate": success_rate,
                    "connection_stats": stats,
                },
                "message": f"Persistent connection test: {operations_successful}/{total_operations} operations successful",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_keep_alive_integration(self) -> Dict[str, Any]:
        """Test keep-alive integration with all features"""
        try:
            logger.info("Testing keep-alive integration...")

            keep_alive = ConnectionKeepAlive(self.ssid, is_demo=True)

            # Event tracking
            events_received = []

            async def track_events(event_type):
                def handler(data):
                    events_received.append(
                        {"type": event_type, "time": datetime.now(), "data": data}
                    )

                return handler

            # Add event handlers
            keep_alive.add_event_handler("connected", await track_events("connected"))
            keep_alive.add_event_handler(
                "message_received", await track_events("message")
            )
            keep_alive.add_event_handler(
                "reconnected", await track_events("reconnected")
            )

            # Start connection
            success = await keep_alive.start_persistent_connection()
            if not success:
                return {
                    "success": False,
                    "error": "Failed to start keep-alive connection",
                }

            # Let it run and test messaging
            await asyncio.sleep(5)

            # Send test messages
            messages_sent = 0
            for i in range(10):
                success = await keep_alive.send_message('42["ps"]')
                if success:
                    messages_sent += 1
                await asyncio.sleep(0.5)

            # Get statistics
            stats = keep_alive.get_connection_stats()

            await keep_alive.stop_persistent_connection()

            # Analyze results
            connected_events = [e for e in events_received if e["type"] == "connected"]
            message_events = [e for e in events_received if e["type"] == "message"]

            return {
                "success": len(connected_events) > 0
                and messages_sent > 8,  # Most messages should succeed
                "details": {
                    "connected_events": len(connected_events),
                    "message_events": len(message_events),
                    "messages_sent": messages_sent,
                    "connection_stats": stats,
                    "total_events": len(events_received),
                },
                "message": f"Keep-alive test: {len(connected_events)} connections, {messages_sent} messages sent, {len(message_events)} messages received",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_monitoring_integration(self) -> Dict[str, Any]:
        """Test monitoring integration"""
        try:
            logger.info("Testing monitoring integration...")

            monitor = ConnectionMonitor(self.ssid, is_demo=True)

            # Start monitoring
            success = await monitor.start_monitoring(persistent_connection=True)
            if not success:
                return {"success": False, "error": "Failed to start monitoring"}

            # Let monitoring run
            await asyncio.sleep(10)

            # Get stats and generate report
            stats = monitor.get_real_time_stats()
            historical = monitor.get_historical_metrics(hours=1)
            report = monitor.generate_diagnostics_report()

            await monitor.stop_monitoring()

            return {
                "success": stats["is_connected"] and stats["total_messages"] > 0,
                "details": {
                    "real_time_stats": stats,
                    "historical_metrics_count": historical["connection_metrics_count"],
                    "health_score": report["health_score"],
                    "health_status": report["health_status"],
                },
                "message": f"Monitoring test: Health score {report['health_score']}/100, {stats['total_messages']} messages monitored",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_multi_client_scenarios(self) -> Dict[str, Any]:
        """Test multiple clients working simultaneously"""
        try:
            logger.info("Testing multi-client scenarios...")

            clients = []

            # Create multiple clients
            for i in range(3):
                client = AsyncPocketOptionClient(
                    self.ssid,
                    is_demo=True,
                    persistent_connection=(i % 2 == 0),  # Mix of persistent and regular
                )
                clients.append(client)

            # Connect all clients
            connect_tasks = [client.connect() for client in clients]
            connect_results = await asyncio.gather(
                *connect_tasks, return_exceptions=True
            )

            successful_connections = sum(1 for r in connect_results if r is True)

            # Run operations on all connected clients
            async def client_operations(client, client_id):
                operations = 0
                try:
                    for _ in range(5):
                        balance = await client.get_balance()
                        if balance:
                            operations += 1
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Client {client_id} operation failed: {e}")
                return operations

            # Run operations concurrently
            operation_tasks = [
                client_operations(client, i)
                for i, client in enumerate(clients)
                if connect_results[i] is True
            ]

            if operation_tasks:
                operation_results = await asyncio.gather(
                    *operation_tasks, return_exceptions=True
                )
                total_operations = sum(
                    r for r in operation_results if isinstance(r, int)
                )
            else:
                total_operations = 0

            # Cleanup
            disconnect_tasks = [
                client.disconnect() for client in clients if client.is_connected
            ]
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            return {
                "success": successful_connections >= 2
                and total_operations > 10,  # At least 2 clients, 10+ operations
                "details": {
                    "total_clients": len(clients),
                    "successful_connections": successful_connections,
                    "total_operations": total_operations,
                    "avg_operations_per_client": total_operations
                    / max(successful_connections, 1),
                },
                "message": f"Multi-client test: {successful_connections}/{len(clients)} clients connected, {total_operations} total operations",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms"""
        try:
            logger.info("Testing error recovery...")

            client = AsyncPocketOptionClient(
                self.ssid, is_demo=True, auto_reconnect=True
            )

            # Connect
            success = await client.connect()
            if not success:
                return {"success": False, "error": "Initial connection failed"}

            # Test graceful handling of invalid operations
            error_scenarios = []

            # Test 1: Invalid asset
            try:
                await client.get_candles("INVALID_ASSET", TimeFrame.M1, 10)
                error_scenarios.append({"test": "invalid_asset", "handled": False})
            except Exception:
                error_scenarios.append({"test": "invalid_asset", "handled": True})

            # Test 2: Invalid timeframe
            try:
                await client.get_candles("EURUSD", "INVALID_TIMEFRAME", 10)
                error_scenarios.append({"test": "invalid_timeframe", "handled": False})
            except Exception:
                error_scenarios.append({"test": "invalid_timeframe", "handled": True})

            # Test 3: Connection still works after errors
            try:
                balance = await client.get_balance()
                connection_recovered = balance is not None
            except Exception:
                connection_recovered = False

            await client.disconnect()

            errors_handled = sum(
                1 for scenario in error_scenarios if scenario["handled"]
            )

            return {
                "success": errors_handled >= 2 and connection_recovered,
                "details": {
                    "error_scenarios": error_scenarios,
                    "errors_handled": errors_handled,
                    "connection_recovered": connection_recovered,
                },
                "message": f"Error recovery test: {errors_handled}/{len(error_scenarios)} errors handled gracefully, connection recovery: {connection_recovered}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_performance_integration(self) -> Dict[str, Any]:
        """Test performance under integrated load"""
        try:
            logger.info("Testing performance integration...")

            # Use load tester for performance testing
            load_tester = LoadTester(self.ssid, is_demo=True)

            config = LoadTestConfig(
                concurrent_clients=2,
                operations_per_client=5,
                operation_delay=0.5,
                use_persistent_connection=True,
                stress_mode=False,
            )

            report = await load_tester.run_load_test(config)

            summary = report["test_summary"]

            # Performance thresholds
            good_throughput = summary["avg_operations_per_second"] > 1.0
            good_success_rate = summary["success_rate"] > 0.9
            reasonable_duration = summary["total_duration"] < 30.0

            return {
                "success": good_throughput
                and good_success_rate
                and reasonable_duration,
                "details": {
                    "throughput": summary["avg_operations_per_second"],
                    "success_rate": summary["success_rate"],
                    "duration": summary["total_duration"],
                    "total_operations": summary["total_operations"],
                },
                "message": f"Performance test: {summary['avg_operations_per_second']:.1f} ops/sec, {summary['success_rate']:.1%} success rate",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_data_consistency_integration(self) -> Dict[str, Any]:
        """Test data consistency across different connection types"""
        try:
            logger.info("Testing data consistency...")

            # Get data from different client types
            data_sources = {}

            # Regular client
            client1 = AsyncPocketOptionClient(self.ssid, is_demo=True)
            success = await client1.connect()
            if success:
                balance1 = await client1.get_balance()
                candles1 = await client1.get_candles("EURUSD", TimeFrame.M1, 5)
                data_sources["regular"] = {
                    "balance": balance1.balance if balance1 else None,
                    "candles_count": len(candles1),
                    "latest_candle": candles1[-1].close if candles1 else None,
                }
                await client1.disconnect()

            # Persistent client
            client2 = AsyncPocketOptionClient(
                self.ssid, is_demo=True, persistent_connection=True
            )
            success = await client2.connect()
            if success:
                balance2 = await client2.get_balance()
                candles2 = await client2.get_candles("EURUSD", TimeFrame.M1, 5)
                data_sources["persistent"] = {
                    "balance": balance2.balance if balance2 else None,
                    "candles_count": len(candles2),
                    "latest_candle": candles2[-1].close if candles2 else None,
                }
                await client2.disconnect()

            # Compare data consistency
            consistency_checks = []

            if "regular" in data_sources and "persistent" in data_sources:
                # Balance should be the same (allowing for small differences due to timing)
                balance_diff = (
                    abs(
                        data_sources["regular"]["balance"]
                        - data_sources["persistent"]["balance"]
                    )
                    if data_sources["regular"]["balance"]
                    and data_sources["persistent"]["balance"]
                    else 0
                )
                consistency_checks.append(
                    {
                        "check": "balance_consistency",
                        "consistent": balance_diff < 0.01,  # Allow 1 cent difference
                    }
                )

                # Candle count should be the same
                consistency_checks.append(
                    {
                        "check": "candles_count_consistency",
                        "consistent": data_sources["regular"]["candles_count"]
                        == data_sources["persistent"]["candles_count"],
                    }
                )

            consistent_checks = sum(
                1 for check in consistency_checks if check["consistent"]
            )

            return {
                "success": consistent_checks
                >= len(consistency_checks) * 0.8,  # 80% consistency
                "details": {
                    "data_sources": data_sources,
                    "consistency_checks": consistency_checks,
                    "consistent_checks": consistent_checks,
                    "total_checks": len(consistency_checks),
                },
                "message": f"Data consistency test: {consistent_checks}/{len(consistency_checks)} checks passed",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_long_running_stability(self) -> Dict[str, Any]:
        """Test stability over extended period"""
        try:
            logger.info("Testing long-running stability...")

            client = AsyncPocketOptionClient(
                self.ssid, is_demo=True, persistent_connection=True, auto_reconnect=True
            )

            success = await client.connect()
            if not success:
                return {
                    "success": False,
                    "error": "Failed to connect for stability test",
                }

            # Track operations over time
            operations_log = []
            start_time = datetime.now()

            # Run for 60 seconds
            while (datetime.now() - start_time).total_seconds() < 60:
                try:
                    # Perform operation
                    balance = await client.get_balance()
                    operations_log.append(
                        {
                            "time": datetime.now(),
                            "success": balance is not None,
                            "operation": "get_balance",
                        }
                    )

                    # Send ping
                    ping_success = await client.send_message('42["ps"]')
                    operations_log.append(
                        {
                            "time": datetime.now(),
                            "success": ping_success,
                            "operation": "ping",
                        }
                    )

                    await asyncio.sleep(2)  # Operation every 2 seconds

                except Exception as e:
                    operations_log.append(
                        {
                            "time": datetime.now(),
                            "success": False,
                            "operation": "error",
                            "error": str(e),
                        }
                    )

            await client.disconnect()

            # Analyze stability
            total_operations = len(operations_log)
            successful_operations = sum(1 for op in operations_log if op["success"])
            success_rate = (
                successful_operations / total_operations if total_operations > 0 else 0
            )

            # Check for any major gaps in operations
            time_gaps = []
            for i in range(1, len(operations_log)):
                gap = (
                    operations_log[i]["time"] - operations_log[i - 1]["time"]
                ).total_seconds()
                if gap > 10:  # More than 10 seconds gap
                    time_gaps.append(gap)

            return {
                "success": success_rate > 0.9
                and len(time_gaps) == 0,  # 90% success rate, no major gaps
                "details": {
                    "total_operations": total_operations,
                    "successful_operations": successful_operations,
                    "success_rate": success_rate,
                    "time_gaps": time_gaps,
                    "duration_seconds": 60,
                },
                "message": f"Stability test: {successful_operations}/{total_operations} operations successful ({success_rate:.1%}), {len(time_gaps)} gaps detected",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration test report"""

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

        total_duration = (datetime.now() - self.start_time).total_seconds()
        test_duration = sum(
            result.get("duration", 0) for result in self.test_results.values()
        )

        # Calculate overall system health score
        health_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # Generate recommendations
        recommendations = []

        if health_score < 80:
            recommendations.append(
                "System health below 80%. Review failed tests and address issues."
            )

        if failed_tests > 0:
            failed_test_names = [
                name
                for name, result in self.test_results.items()
                if result["status"] == "FAILED"
            ]
            recommendations.append(
                f"Failed tests need attention: {', '.join(failed_test_names)}"
            )

        if error_tests > 0:
            recommendations.append(
                "Some tests encountered errors. Check logs for details."
            )

        if health_score >= 90:
            recommendations.append(
                "Excellent system health! All major components working well."
            )
        elif health_score >= 80:
            recommendations.append("Good system health with minor issues to address.")

        report = {
            "integration_summary": {
                "test_start_time": self.start_time.isoformat(),
                "test_end_time": datetime.now().isoformat(),
                "total_duration": total_duration,
                "test_execution_time": test_duration,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "health_score": health_score,
                "health_status": (
                    "EXCELLENT"
                    if health_score >= 90
                    else "GOOD"
                    if health_score >= 80
                    else "FAIR"
                    if health_score >= 60
                    else "POOR"
                ),
            },
            "detailed_results": self.test_results,
            "recommendations": recommendations,
            "system_assessment": {
                "connectivity": self._assess_connectivity(),
                "performance": self._assess_performance(),
                "reliability": self._assess_reliability(),
                "monitoring": self._assess_monitoring(),
            },
        }

        return report

    def _assess_connectivity(self) -> Dict[str, Any]:
        """Assess connectivity aspects"""
        connectivity_tests = [
            "Basic Connectivity",
            "SSID Format Compatibility",
            "Persistent Connection Integration",
        ]
        passed = sum(
            1
            for test in connectivity_tests
            if self.test_results.get(test, {}).get("status") == "PASSED"
        )

        return {
            "score": (passed / len(connectivity_tests)) * 100,
            "status": "GOOD"
            if passed >= len(connectivity_tests) * 0.8
            else "NEEDS_ATTENTION",
            "details": f"{passed}/{len(connectivity_tests)} connectivity tests passed",
        }

    def _assess_performance(self) -> Dict[str, Any]:
        """Assess performance aspects"""
        performance_tests = [
            "Performance Under Load",
            "Long-Running Stability",
            "Multi-Client Scenarios",
        ]
        passed = sum(
            1
            for test in performance_tests
            if self.test_results.get(test, {}).get("status") == "PASSED"
        )

        return {
            "score": (passed / len(performance_tests)) * 100,
            "status": "GOOD"
            if passed >= len(performance_tests) * 0.8
            else "NEEDS_ATTENTION",
            "details": f"{passed}/{len(performance_tests)} performance tests passed",
        }

    def _assess_reliability(self) -> Dict[str, Any]:
        """Assess reliability aspects"""
        reliability_tests = [
            "Error Recovery",
            "Keep-Alive Functionality",
            "Data Consistency",
        ]
        passed = sum(
            1
            for test in reliability_tests
            if self.test_results.get(test, {}).get("status") == "PASSED"
        )

        return {
            "score": (passed / len(reliability_tests)) * 100,
            "status": "GOOD"
            if passed >= len(reliability_tests) * 0.8
            else "NEEDS_ATTENTION",
            "details": f"{passed}/{len(reliability_tests)} reliability tests passed",
        }

    def _assess_monitoring(self) -> Dict[str, Any]:
        """Assess monitoring aspects"""
        monitoring_tests = ["Monitoring Integration"]
        passed = sum(
            1
            for test in monitoring_tests
            if self.test_results.get(test, {}).get("status") == "PASSED"
        )

        return {
            "score": (passed / len(monitoring_tests)) * 100
            if len(monitoring_tests) > 0
            else 100,
            "status": "GOOD"
            if passed >= len(monitoring_tests) * 0.8
            else "NEEDS_ATTENTION",
            "details": f"{passed}/{len(monitoring_tests)} monitoring tests passed",
        }


async def run_integration_tests(ssid: str = None):
    """Run the full integration test suite"""

    if not ssid:
        ssid = r'42["auth",{"session":"integration_test_session","isDemo":1,"uid":0,"platform":1}]'
        logger.warning("Using demo SSID for integration testing")

    logger.info("PocketOption API Integration Testing Suite")
    logger.info("=" * 60)
    logger.info("This comprehensive test validates all components working together")
    logger.info("")

    tester = IntegrationTester(ssid)

    try:
        report = await tester.run_full_integration_tests()

        # Print comprehensive summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)

        summary = report["integration_summary"]
        logger.info(f"Tests Executed: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']} ")
        logger.info(f"Failed: {summary['failed_tests']} ❌")
        logger.info(f"Errors: {summary['error_tests']} 💥")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(
            f"Health Score: {summary['health_score']:.1f}/100 ({summary['health_status']})"
        )
        logger.info(f"Total Duration: {summary['total_duration']:.2f}s")

        # System assessment
        logger.info("\n📋 SYSTEM ASSESSMENT")
        logger.info("-" * 30)
        assessment = report["system_assessment"]
        for aspect, details in assessment.items():
            status_emoji = "" if details["status"] == "GOOD" else "⚠️"
            logger.info(
                f"{status_emoji} {aspect.title()}: {details['score']:.0f}/100 - {details['details']}"
            )

        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS")
        logger.info("-" * 30)
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"{i}. {rec}")

        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"integration_test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n📄 Detailed report saved to: {report_file}")

        # Final verdict
        if summary["health_score"] >= 90:
            logger.success("🎉 EXCELLENT: System is performing exceptionally well!")
        elif summary["health_score"] >= 80:
            logger.info(
                "👍 GOOD: System is performing well with minor areas for improvement"
            )
        elif summary["health_score"] >= 60:
            logger.warning("FAIR: System has some issues that should be addressed")
        else:
            logger.error(
                "POOR: System has significant issues requiring immediate attention"
            )

        return report

    except Exception as e:
        logger.error(f"Integration testing failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    # Allow passing SSID as command line argument
    ssid = None
    if len(sys.argv) > 1:
        ssid = sys.argv[1]
        logger.info(f"Using provided SSID: {ssid[:50]}...")

    asyncio.run(run_integration_tests(ssid))
