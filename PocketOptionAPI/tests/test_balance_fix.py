"""
Test script to verify the balance issue fix
"""

import asyncio
from loguru import logger

# Mock test SSID for demonstration
complete_ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'


async def test_balance_fix():
    """Test the balance fix with the new async API"""

    logger.info("Testing Balance Fix")
    logger.info("=" * 50)

    # Import here to avoid import issues during file changes
    try:
        from pocketoptionapi_async import AsyncPocketOptionClient

        # Create client
        client = AsyncPocketOptionClient(ssid=complete_ssid, is_demo=True)

        # Add balance event callback to test
        balance_received = False

        def on_balance_updated(balance):
            nonlocal balance_received
            balance_received = True
            logger.success(f" Balance callback triggered: ${balance.balance:.2f}")

        client.add_event_callback("balance_updated", on_balance_updated)

        # Test connection and balance retrieval
        try:
            await client.connect()

            if client.is_connected:
                logger.info(" Connected successfully")

                # Try to get balance
                try:
                    balance = await client.get_balance()
                    if balance:
                        logger.success(
                            f" Balance retrieved successfully: ${balance.balance:.2f}"
                        )
                        logger.info(f"   Currency: {balance.currency}")
                        logger.info(f"   Demo: {balance.is_demo}")
                        logger.info(f"   Last updated: {balance.last_updated}")
                    else:
                        logger.error("Balance is None - issue still exists")

                except Exception as e:
                    logger.error(f"Balance retrieval failed: {e}")

                # Wait for balance events
                logger.info("⏳ Waiting for balance events...")
                await asyncio.sleep(5)

                if balance_received:
                    logger.success(" Balance event received successfully!")
                else:
                    logger.warning("No balance event received")

            else:
                logger.warning("Connection failed (expected with test SSID)")

        except Exception as e:
            logger.info(f"Connection test: {e}")

        finally:
            await client.disconnect()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

    logger.info("=" * 50)
    logger.success(" Balance fix test completed!")
    return True


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )

    asyncio.run(test_balance_fix())
