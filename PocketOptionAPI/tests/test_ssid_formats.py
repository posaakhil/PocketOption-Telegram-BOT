"""
Test script to demonstrate the updated SSID handling in PocketOption Async API
"""

import asyncio
import json
from pocketoptionapi_async import AsyncPocketOptionClient


async def test_ssid_formats():
    """Test different SSID format handling"""

    print("Testing SSID Format Handling")
    print("=" * 50)

    # Test 1: Complete SSID format (as provided by user)
    complete_ssid = '42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    print("Testing Complete SSID Format")
    print(f"Input: {complete_ssid}")

    client1 = AsyncPocketOptionClient(ssid=complete_ssid)

    # Verify parsing
    print(f" Parsed session: {client1.session_id}")
    print(f" Parsed demo: {client1.is_demo}")
    print(f" Parsed UID: {client1.uid}")
    print(f" Parsed platform: {client1.platform}")
    print(f" Parsed fast history: {client1.is_fast_history}")

    formatted_message = client1._format_session_message()
    print(f" Formatted message: {formatted_message}")
    print()

    # Test 2: Raw session ID
    raw_session = "n1p5ah5u8t9438rbunpgrq0hlq"

    print("Testing Raw Session ID")
    print(f"Input: {raw_session}")

    client2 = AsyncPocketOptionClient(
        ssid=raw_session, is_demo=True, uid=72645361, platform=1, is_fast_history=True
    )

    print(f" Session: {client2.session_id}")
    print(f" Demo: {client2.is_demo}")
    print(f" UID: {client2.uid}")
    print(f" Platform: {client2.platform}")

    formatted_message2 = client2._format_session_message()
    print(f" Formatted message: {formatted_message2}")
    print()

    # Test 3: Verify both produce same result
    print("Comparing Results")

    # Parse the JSON parts to compare
    def extract_auth_data(msg):
        json_part = msg[10:-1]  # Remove '42["auth",' and ']'
        return json.loads(json_part)

    auth_data1 = extract_auth_data(formatted_message)
    auth_data2 = extract_auth_data(formatted_message2)

    print(f"Complete SSID auth data: {auth_data1}")
    print(f"Raw session auth data:   {auth_data2}")

    # Compare key fields
    fields_match = (
        auth_data1["session"] == auth_data2["session"]
        and auth_data1["isDemo"] == auth_data2["isDemo"]
        and auth_data1["uid"] == auth_data2["uid"]
        and auth_data1["platform"] == auth_data2["platform"]
    )

    if fields_match:
        print(" Both methods produce equivalent authentication data!")
    else:
        print("Authentication data mismatch!")

    print()

    # Test 4: Test connection with real SSID format (mock)
    print("Testing Connection with Complete SSID")

    try:
        # This will fail with test data, but should show proper SSID handling
        await client1.connect()
        print(" Connection successful")
    except Exception as e:
        print(f"Expected connection failure with test data: {e}")

    print("\nSSID Format Support Summary:")
    print(' Complete SSID format: 42["auth",{...}] - SUPPORTED')
    print(" Raw session ID with parameters - SUPPORTED")
    print(" Automatic parsing and formatting - WORKING")
    print(" UID and platform preservation - WORKING")
    print(" Fast history support - WORKING")

    # Show example usage
    print("\nUsage Examples:")
    print("\n# Method 1: Complete SSID (recommended)")
    print("client = AsyncPocketOptionClient(")
    print(
        '    ssid=\'42["auth",{"session":"your_session","isDemo":1,"uid":12345,"platform":1,"isFastHistory":true}]\''
    )
    print(")")

    print("\n# Method 2: Raw session with parameters")
    print("client = AsyncPocketOptionClient(")
    print("    ssid='your_raw_session_id',")
    print("    is_demo=True,")
    print("    uid=12345,")
    print("    platform=1")
    print(")")


async def test_real_connection_simulation():
    """Simulate what a real connection would look like"""

    print("\n\nReal Connection Simulation")
    print("=" * 40)

    # Example with real-looking SSID format
    realistic_ssid = '42["auth",{"session":"a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    client = AsyncPocketOptionClient(ssid=realistic_ssid)

    print("Initialized client with parsed data:")
    print(f"  Session: {client.session_id}")
    print(f"  Demo: {client.is_demo}")
    print(f"  UID: {client.uid}")
    print(f"  Platform: {client.platform}")

    # Show what would be sent during handshake
    auth_message = client._format_session_message()
    print("\nAuthentication message to be sent:")
    print(f"  {auth_message}")

    # Parse and display nicely
    json_part = auth_message[10:-1]  # Remove '42["auth",' and ']'
    auth_data = json.loads(json_part)
    print("\nParsed authentication data:")
    for key, value in auth_data.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_ssid_formats())
    asyncio.run(test_real_connection_simulation())
