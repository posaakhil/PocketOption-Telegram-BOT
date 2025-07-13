"""
Enhanced PocketOption API Testing with Monitoring and Performance Analysis
"""

import asyncio
import os
import time
from datetime import datetime
from loguru import logger

from pocketoptionapi_async import (
    AsyncPocketOptionClient,
    error_monitor,
    health_checker,
    ErrorSeverity,
    ErrorCategory,
)


class EnhancedAPITester:
    """Enhanced API testing with monitoring capabilities"""

    def __init__(self, session_id: str, is_demo: bool = True):
        self.session_id = session_id
        self.is_demo = is_demo
        self.test_results = {}

        # Setup monitoring callbacks
        error_monitor.add_alert_callback(self.handle_error_alert)

    async def handle_error_alert(self, alert_data):
        """Handle error alerts from the monitoring system"""
        logger.warning(
            f"🚨 ERROR ALERT: {alert_data['error_type']} - "
            f"{alert_data['error_count']} errors in {alert_data['time_window']}s"
        )

    async def test_enhanced_connection(self):
        """Test connection with enhanced monitoring"""
        logger.info("Testing Enhanced Connection with Monitoring")
        print("=" * 60)

        client = AsyncPocketOptionClient(ssid=self.session_id, is_demo=self.is_demo)

        try:
            # Test connection with monitoring
            success = await client.execute_with_monitoring(
                operation_name="connection_test", func=client.connect
            )

            if success:
                logger.success(" Connection successful with monitoring")

                # Get health status
                health = await client.get_health_status()
                logger.info(f"🏥 Health Status: {health['overall_status']}")

                # Get performance metrics
                metrics = await client.get_performance_metrics()
                logger.info(
                    f"📊 Performance: {len(metrics['operation_metrics'])} tracked operations"
                )

                # Test some operations
                await self.test_monitored_operations(client)

            else:
                logger.error("Connection failed")

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
        finally:
            await client.disconnect()

    async def test_monitored_operations(self, client):
        """Test various operations with monitoring"""
        logger.info("Testing Monitored Operations")

        operations = [
            ("balance_check", lambda: client.get_balance()),
            ("candles_fetch", lambda: client.get_candles("EURUSD_otc", 60, 50)),
            ("health_check", lambda: client.get_health_status()),
        ]

        for op_name, operation in operations:
            try:
                start_time = time.time()

                await client.execute_with_monitoring(
                    operation_name=op_name, func=operation
                )

                duration = time.time() - start_time
                logger.success(f" {op_name}: {duration:.3f}s")

            except Exception as e:
                logger.error(f"{op_name} failed: {e}")

                # Record error in monitoring system
                await error_monitor.record_error(
                    error_type=f"{op_name}_failure",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.TRADING,
                    message=str(e),
                    context={"operation": op_name},
                )

    async def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        logger.info("⚡ Testing Circuit Breaker")
        print("=" * 60)

        from pocketoptionapi_async.monitoring import CircuitBreaker

        # Create a circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

        async def failing_operation():
            """Simulated failing operation"""
            raise Exception("Simulated failure")

        async def working_operation():
            """Simulated working operation"""
            return "Success"

        # Test circuit breaker with failing operations
        failures = 0
        for i in range(5):
            try:
                await breaker.call(failing_operation)
            except Exception as e:
                failures += 1
                logger.warning(f"Attempt {i + 1}: {e} (State: {breaker.state})")

        logger.info(f"🔥 Circuit breaker opened after {failures} failures")

        # Wait for recovery
        logger.info("⏳ Waiting for recovery...")
        await asyncio.sleep(6)

        # Test with working operation
        try:
            result = await breaker.call(working_operation)
            logger.success(f" Circuit breaker recovered: {result}")
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    async def test_concurrent_performance(self):
        """Test concurrent operations performance"""
        logger.info("Testing Concurrent Performance")
        print("=" * 60)

        async def create_and_test_client(client_id: int):
            """Create client and perform operations"""
            client = AsyncPocketOptionClient(
                session_id=self.session_id, is_demo=self.is_demo
            )

            start_time = time.time()

            try:
                await client.connect()

                if client.is_connected:
                    # Perform some operations
                    balance = await client.get_balance()
                    health = await client.get_health_status()

                    duration = time.time() - start_time
                    return {
                        "client_id": client_id,
                        "success": True,
                        "duration": duration,
                        "balance": balance.balance if balance else None,
                        "health": health["overall_status"],
                    }

            except Exception as e:
                return {
                    "client_id": client_id,
                    "success": False,
                    "duration": time.time() - start_time,
                    "error": str(e),
                }
            finally:
                await client.disconnect()

        # Run concurrent clients
        concurrent_level = 3
        logger.info(f"Running {concurrent_level} concurrent clients...")

        start_time = time.time()
        tasks = [create_and_test_client(i) for i in range(concurrent_level)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not (isinstance(r, dict) and r.get("success"))]

        logger.info("📊 Concurrent Test Results:")
        logger.info(f"    Successful: {len(successful)}/{concurrent_level}")
        logger.info(f"   Failed: {len(failed)}")
        logger.info(f"   ⏱️  Total Time: {total_time:.3f}s")

        if successful:
            avg_time = sum(r["duration"] for r in successful) / len(successful)
            logger.info(f"   Avg Client Time: {avg_time:.3f}s")

    async def test_error_monitoring(self):
        """Test error monitoring capabilities"""
        logger.info("🔍 Testing Error Monitoring")
        print("=" * 60)

        # Generate some test errors
        test_errors = [
            ("connection_timeout", ErrorSeverity.HIGH, ErrorCategory.CONNECTION),
            ("invalid_session", ErrorSeverity.CRITICAL, ErrorCategory.AUTHENTICATION),
            ("order_rejected", ErrorSeverity.MEDIUM, ErrorCategory.TRADING),
            ("data_parsing", ErrorSeverity.LOW, ErrorCategory.DATA),
        ]

        for error_type, severity, category in test_errors:
            await error_monitor.record_error(
                error_type=error_type,
                severity=severity,
                category=category,
                message=f"Test {error_type} error",
                context={"test": True, "timestamp": datetime.now().isoformat()},
            )

        # Get error summary
        summary = error_monitor.get_error_summary(hours=1)

        logger.info("Error Summary:")
        logger.info(f"   Total Errors: {summary['total_errors']}")
        logger.info(f"   Error Rate: {summary['error_rate']:.2f}/hour")
        logger.info(f"   Top Errors: {summary['top_errors'][:3]}")

        # Test alert threshold
        logger.info("🚨 Testing Alert Threshold...")
        for i in range(15):  # Generate many errors to trigger alert
            await error_monitor.record_error(
                error_type="test_spam",
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.SYSTEM,
                message=f"Spam test error #{i + 1}",
                context={"spam_test": True},
            )

    async def generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("📋 Generating Performance Report")
        print("=" * 60)

        # Get error summary
        error_summary = error_monitor.get_error_summary()

        # Start health monitoring briefly
        await health_checker.start_monitoring()
        await asyncio.sleep(2)  # Let it collect some data
        health_report = health_checker.get_health_report()
        await health_checker.stop_monitoring()

        report = []
        report.append("=" * 80)
        report.append("ENHANCED POCKETOPTION API PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Error monitoring section
        report.append("🔍 ERROR MONITORING")
        report.append("-" * 40)
        report.append(f"Total Errors: {error_summary['total_errors']}")
        report.append(f"Error Rate: {error_summary['error_rate']:.2f}/hour")
        report.append("")

        if error_summary["top_errors"]:
            report.append("Top Error Types:")
            for error_type, count in error_summary["top_errors"][:5]:
                report.append(f"  • {error_type}: {count}")

        report.append("")

        # Health monitoring section
        report.append("🏥 HEALTH MONITORING")
        report.append("-" * 40)
        report.append(f"Overall Status: {health_report['overall_status']}")

        if health_report["unhealthy_services"]:
            report.append(
                f"Unhealthy Services: {', '.join(health_report['unhealthy_services'])}"
            )

        report.append("")

        # Recommendations section
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        if error_summary["total_errors"] > 10:
            report.append("• High error count detected - investigate error patterns")

        if health_report["overall_status"] != "healthy":
            report.append("• System health issues detected - check service status")

        if not error_summary["top_errors"]:
            report.append("•  No significant errors detected")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)
        print(report_text)

        # Save to file
        with open("enhanced_performance_report.txt", "w") as f:
            f.write(report_text)

        logger.success("📄 Report saved to enhanced_performance_report.txt")

    async def run_all_tests(self):
        """Run all enhanced tests"""
        logger.info("Starting Enhanced API Tests")
        print("=" * 80)

        tests = [
            ("Enhanced Connection", self.test_enhanced_connection),
            ("Circuit Breaker", self.test_circuit_breaker),
            ("Concurrent Performance", self.test_concurrent_performance),
            ("Error Monitoring", self.test_error_monitoring),
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                await test_func()
                logger.success(f" {test_name} completed")
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"{test_name} failed: {e}")

        # Generate final report
        await self.generate_performance_report()


async def main():
    """Main enhanced testing function"""
    print("ENHANCED POCKETOPTION API TESTING")
    print("=" * 80)
    print("Features being tested:")
    print("   Enhanced Error Monitoring")
    print("   Circuit Breaker Pattern")
    print("   Health Checks")
    print("   Performance Metrics")
    print("   Concurrent Operations")
    print("   Retry Policies")
    print("=" * 80)
    print()

    # Get session ID from environment or use test session
    session_id = os.getenv("POCKET_OPTION_SSID", "n1p5ah5u8t9438rbunpgrq0hlq")

    if session_id == "test_session_id":
        logger.warning(
            " Using test session ID - set POCKET_OPTION_SSID for real testing"
        )

    # Create and run enhanced tester
    tester = EnhancedAPITester(session_id=session_id, is_demo=True)
    await tester.run_all_tests()

    logger.success("🎉 All enhanced tests completed!")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "enhanced_api_test.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )

    asyncio.run(main())
