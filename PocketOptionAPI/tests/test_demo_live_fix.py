"""
Test script to verify the demo/live mode fix
"""

import asyncio
import json
from pocketoptionapi_async import AsyncPocketOptionClient


async def test_demo_live_fix():
    """Test that is_demo parameter is properly respected"""

    # Test SSID with demo=1 hardcoded (should be overridden by is_demo parameter)
    demo_ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    print("Testing Demo/Live Mode Fix")
    print("=" * 50)

    # Test 1: Demo mode with demo SSID (should work)
    print("\nTest: is_demo=True with demo SSID")
    client_demo = AsyncPocketOptionClient(ssid=demo_ssid, is_demo=True)
    formatted_demo = client_demo._format_session_message()
    parsed_demo = json.loads(formatted_demo[10:-1])  # Extract JSON part

    print(f"   SSID isDemo value: {json.loads(demo_ssid[10:-1])['isDemo']}")
    print("   Constructor is_demo: True")
    print(f"   Client is_demo: {client_demo.is_demo}")
    print(f"   Formatted message isDemo: {parsed_demo['isDemo']}")
    print(f"    Expected: 1, Got: {parsed_demo['isDemo']}")

    # Test 2: Live mode with demo SSID (should override to live)
    print("\nTest: is_demo=False with demo SSID")
    client_live = AsyncPocketOptionClient(ssid=demo_ssid, is_demo=False)
    formatted_live = client_live._format_session_message()
    parsed_live = json.loads(formatted_live[10:-1])  # Extract JSON part

    print(f"   SSID isDemo value: {json.loads(demo_ssid[10:-1])['isDemo']}")
    print("   Constructor is_demo: False")
    print(f"   Client is_demo: {client_live.is_demo}")
    print(f"   Formatted message isDemo: {parsed_live['isDemo']}")
    print(f"    Expected: 0, Got: {parsed_live['isDemo']}")

    # Test 3: Raw session ID with demo mode
    print("\nTest: Raw session with is_demo=True")
    raw_session = "n1p5ah5u8t9438rbunpgrq0hlq"
    client_raw_demo = AsyncPocketOptionClient(
        ssid=raw_session, is_demo=True, uid=72645361
    )
    formatted_raw_demo = client_raw_demo._format_session_message()
    parsed_raw_demo = json.loads(formatted_raw_demo[10:-1])

    print("   Constructor is_demo: True")
    print(f"   Client is_demo: {client_raw_demo.is_demo}")
    print(f"   Formatted message isDemo: {parsed_raw_demo['isDemo']}")
    print(f"    Expected: 1, Got: {parsed_raw_demo['isDemo']}")

    # Test 4: Raw session ID with live mode
    print("\nTest: Raw session with is_demo=False")
    client_raw_live = AsyncPocketOptionClient(
        ssid=raw_session, is_demo=False, uid=72645361
    )
    formatted_raw_live = client_raw_live._format_session_message()
    parsed_raw_live = json.loads(formatted_raw_live[10:-1])

    print("   Constructor is_demo: False")
    print(f"   Client is_demo: {client_raw_live.is_demo}")
    print(f"   Formatted message isDemo: {parsed_raw_live['isDemo']}")
    print(f"    Expected: 0, Got: {parsed_raw_live['isDemo']}")

    # Test 5: Region selection based on demo mode
    print("\n5️⃣ Test: Region selection logic")

    # Import regions to check the logic
    from pocketoptionapi_async.constants import REGIONS

    all_regions = REGIONS.get_all_regions()
    demo_regions = REGIONS.get_demo_regions()

    print(f"   Total regions: {len(all_regions)}")
    print(f"   Demo regions: {len(demo_regions)}")

    # Check demo client region selection
    print("\n   Demo client (is_demo=True):")
    demo_region_names = [
        name for name, url in all_regions.items() if url in demo_regions
    ]
    print(f"   Should use demo regions: {demo_region_names}")

    # Check live client region selection
    print("\n   Live client (is_demo=False):")
    live_region_names = [
        name for name, url in all_regions.items() if "DEMO" not in name.upper()
    ]
    print(f"   Should use non-demo regions: {live_region_names}")

    print("\n" + "=" * 50)
    print(" Demo/Live Mode Fix Test Complete!")

    # Verify all tests passed
    demo_test_pass = parsed_demo["isDemo"] == 1
    live_test_pass = parsed_live["isDemo"] == 0
    raw_demo_test_pass = parsed_raw_demo["isDemo"] == 1
    raw_live_test_pass = parsed_raw_live["isDemo"] == 0

    if all([demo_test_pass, live_test_pass, raw_demo_test_pass, raw_live_test_pass]):
        print("🎉 ALL TESTS PASSED! is_demo parameter is now properly respected!")
    else:
        print("Some tests failed. The fix needs adjustment.")

    return all([demo_test_pass, live_test_pass, raw_demo_test_pass, raw_live_test_pass])


if __name__ == "__main__":
    asyncio.run(test_demo_live_fix())
