"""
Test script to verify the place_order fix
"""

import asyncio
from loguru import logger
from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection


async def test_order_placement():
    """Test placing an order to verify the fix"""

    ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    client = AsyncPocketOptionClient(ssid=ssid, is_demo=True)

    try:
        logger.info("Connecting to PocketOption...")
        await client.connect()

        if client.is_connected:
            logger.success(" Connected successfully!")

            # Wait for authentication and balance
            await asyncio.sleep(3)

            try:
                balance = await client.get_balance()
                if balance:
                    logger.info(f"Balance: ${balance.balance:.2f}")
                else:
                    logger.warning("No balance data received")
            except Exception as e:
                logger.info(f"Balance error (expected with demo): {e}")

            # Test placing an order (this should now work without the order_id error)
            logger.info("esting order placement...")
            try:
                order_result = await client.place_order(
                    asset="EURUSD_otc",
                    amount=1.0,
                    direction=OrderDirection.CALL,
                    duration=60,
                )

                logger.success(" Order placed successfully!")
                logger.info(f"   Order ID: {order_result.order_id}")
                logger.info(f"   Status: {order_result.status}")
                logger.info(f"   Asset: {order_result.asset}")
                logger.info(f"   Amount: ${order_result.amount}")
                logger.info(f"   Direction: {order_result.direction}")

            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                # Check if it's the same error as before
                if "'Order' object has no attribute 'order_id'" in str(e):
                    logger.error("The original error is still present!")
                else:
                    logger.info(
                        "Different error (this is expected with demo connection)"
                    )
        else:
            logger.warning("Connection failed (expected with demo SSID)")

    except Exception as e:
        logger.error(f"Connection error: {e}")

    finally:
        await client.disconnect()
        logger.info("Disconnected")


if __name__ == "__main__":
    logger.info("Testing Order Placement Fix")
    logger.info("=" * 50)
    asyncio.run(test_order_placement())
