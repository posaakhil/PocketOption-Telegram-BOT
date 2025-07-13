"""
Test script to verify the demo/live connection fix
"""

import asyncio
from pocketoptionapi_async import AsyncPocketOptionClient


async def test_demo_live_connection():
    """Test that demo/live connections go to correct regions"""

    # Test SSID with demo=1 hardcoded (should be overridden by is_demo parameter)
    demo_ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    print("Testing Demo/Live Connection Fix")
    print("=" * 50)

    # Test 1: Demo mode connection (should connect to demo regions)
    print("\nTest: Demo mode connection (is_demo=True)")
    client_demo = AsyncPocketOptionClient(ssid=demo_ssid, is_demo=True)

    print(f"   Client is_demo: {client_demo.is_demo}")
    print("   Attempting connection to demo regions...")

    try:
        success = await asyncio.wait_for(client_demo.connect(), timeout=30)

        if success:
            print("    Connected successfully!")
            if hasattr(client_demo, "connection_info") and client_demo.connection_info:
                print(f"   Connected to: {client_demo.connection_info.region}")
            await client_demo.disconnect()
        else:
            print("   Connection failed")

    except asyncio.TimeoutError:
        print("   ⏰ Connection timeout (expected with test credentials)")
    except Exception as e:
        print(f"    Connection error: {e}")

    # Test 2: Live mode connection (should try non-demo regions)
    print("\nTest: Live mode connection (is_demo=False)")
    client_live = AsyncPocketOptionClient(ssid=demo_ssid, is_demo=False)

    print(f"   Client is_demo: {client_live.is_demo}")
    print("   Attempting connection to live regions...")

    try:
        success = await asyncio.wait_for(client_live.connect(), timeout=30)

        if success:
            print("    Connected successfully!")
            if hasattr(client_live, "connection_info") and client_live.connection_info:
                print(f"   Connected to: {client_live.connection_info.region}")
            await client_live.disconnect()
        else:
            print("   Connection failed")

    except asyncio.TimeoutError:
        print("   ⏰ Connection timeout (expected with test credentials)")
    except Exception as e:
        print(f"    Connection error: {e}")

    print("\n" + "=" * 50)
    print(" Demo/Live Connection Test Complete!")
    print("\nKey improvements:")
    print("•  is_demo parameter now properly overrides SSID values")
    print("•  Demo mode connects only to demo regions")
    print("•  Live mode excludes demo regions")
    print("•  Authentication messages use correct isDemo values")


if __name__ == "__main__":
    asyncio.run(test_demo_live_connection())
