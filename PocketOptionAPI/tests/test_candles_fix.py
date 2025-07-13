"""
Test script to verify candles data retrieval functionality
"""

import asyncio
import json
from datetime import datetime
from pocketoptionapi_async import AsyncPocketOptionClient


async def test_candles_retrieval():
    """Test candles data retrieval with the fixed implementation"""

    # Replace with your actual SSID
    ssid = "po_session_id=your_session_id_here"

    print("Testing Candles Data Retrieval")
    print("=" * 50)

    try:
        # Create client with logging enabled to see detailed output
        client = AsyncPocketOptionClient(ssid, is_demo=True, enable_logging=True)

        print("📡 Connecting to PocketOption...")
        await client.connect()

        print("\n📊 Requesting candles data...")

        # Test 1: Get recent candles for EURUSD
        asset = "EURUSD"
        timeframe = 60  # 1 minute
        count = 20

        print(f"Asset: {asset}")
        print(f"Timeframe: {timeframe}s (1 minute)")
        print(f"Count: {count}")

        candles = await client.get_candles(asset, timeframe, count)

        if candles:
            print(f"\n Successfully retrieved {len(candles)} candles!")

            # Display first few candles
            print("\nSample candle data:")
            for i, candle in enumerate(candles[:5]):
                print(
                    f"  {i + 1}. {candle.timestamp.strftime('%H:%M:%S')} - "
                    f"O:{candle.open:.5f} H:{candle.high:.5f} L:{candle.low:.5f} C:{candle.close:.5f}"
                )

            if len(candles) > 5:
                print(f"  ... and {len(candles) - 5} more candles")

        else:
            print("No candles received - this may indicate an issue")

        # Test 2: Get candles as DataFrame
        print("\n📊 Testing DataFrame conversion...")
        try:
            df = await client.get_candles_dataframe(asset, timeframe, count)
            if not df.empty:
                print(f" DataFrame created with {len(df)} rows")
                print(f"Columns: {list(df.columns)}")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
            else:
                print("Empty DataFrame received")
        except Exception as e:
            print(f"DataFrame test failed: {e}")

        # Test 3: Different timeframes
        print("\n⏱️ Testing different timeframes...")
        timeframes_to_test = [(60, "1 minute"), (300, "5 minutes"), (900, "15 minutes")]

        for tf_seconds, tf_name in timeframes_to_test:
            try:
                test_candles = await client.get_candles(asset, tf_seconds, 5)
                if test_candles:
                    print(f" {tf_name}: {len(test_candles)} candles")
                else:
                    print(f"{tf_name}: No data")
            except Exception as e:
                print(f"{tf_name}: Error - {e}")

        print("\n🔍 Testing different assets...")
        assets_to_test = ["EURUSD", "GBPUSD", "USDJPY"]

        for test_asset in assets_to_test:
            try:
                test_candles = await client.get_candles(test_asset, 60, 3)
                if test_candles:
                    latest = test_candles[-1] if test_candles else None
                    print(
                        f" {test_asset}: Latest price {latest.close:.5f}"
                        if latest
                        else f" {test_asset}: {len(test_candles)} candles"
                    )
                else:
                    print(f"{test_asset}: No data")
            except Exception as e:
                print(f"{test_asset}: Error - {e}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        try:
            await client.disconnect()
            print("\nDisconnected from PocketOption")
        except:
            pass


async def test_candles_message_format():
    """Test the message format being sent"""

    print("\n🔍 Testing Message Format")
    print("=" * 30)

    # Simulate the message creation
    asset = "EURUSD"
    timeframe = 60
    count = 10
    end_time = datetime.now()
    end_timestamp = int(end_time.timestamp())

    # Create message data in the format expected by PocketOption
    data = {
        "asset": str(asset),
        "index": end_timestamp,
        "offset": count,
        "period": timeframe,
        "time": end_timestamp,
    }

    # Create the full message
    message_data = ["loadHistoryPeriod", data]
    message = f'42["sendMessage",{json.dumps(message_data)}]'

    print(f"Asset: {asset}")
    print(f"Timeframe: {timeframe}s")
    print(f"Count: {count}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timestamp: {end_timestamp}")
    print("\nGenerated message:")
    print(message)
    print("\nMessage data structure:")
    print(json.dumps(message_data, indent=2))


if __name__ == "__main__":
    print("PocketOption Candles Test Suite")
    print("=" * 40)

    # Test message format first
    asyncio.run(test_candles_message_format())

    # Then test actual retrieval (requires valid SSID)
    print("\n" + "=" * 40)
    print(" To test actual candles retrieval:")
    print("1. Replace 'your_session_id_here' with your actual SSID")
    print("2. Uncomment the line below")
    print("=" * 40)

    # Uncomment this line after adding your SSID:
    # asyncio.run(test_candles_retrieval())
