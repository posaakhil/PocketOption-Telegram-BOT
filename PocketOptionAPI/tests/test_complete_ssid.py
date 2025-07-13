"""
Test script demonstrating complete SSID format handling
"""

import asyncio
import os

from pocketoptionapi_async import AsyncPocketOptionClient


async def test_complete_ssid_format():
    """Test the complete SSID format functionality"""

    print("Testing Complete SSID Format Handling")
    print("=" * 50)

    # Test 1: Complete SSID format (what the user wants)
    complete_ssid = r'42["auth",{"session":"n1p5ah5u8t9438rbunpgrq0hlq","isDemo":1,"uid":72645361,"platform":1,"isFastHistory":true}]'

    print("📝 Testing with complete SSID format:")
    print(f"   SSID: {complete_ssid[:50]}...")
    print()

    try:
        # Create client with complete SSID
        client = AsyncPocketOptionClient(ssid=complete_ssid, is_demo=True)

        # Check that the SSID is handled correctly
        formatted_message = client._format_session_message()

        print(" Client created successfully")
        print(f"📤 Formatted message: {formatted_message[:50]}...")
        print(f"🔍 Session extracted: {getattr(client, 'session_id', 'N/A')[:20]}...")
        print(f"👤 UID extracted: {client.uid}")
        print(f"🏷️  Platform: {client.platform}")
        print(f"Demo mode: {client.is_demo}")
        print(f"⚡ Fast history: {client.is_fast_history}")

        # Test connection (will fail with test SSID but should show proper format)
        print("\nTesting connection...")
        try:
            await client.connect()
            if client.is_connected:
                print(" Connected successfully!")
                print(f"📊 Connection info: {client.connection_info}")
            else:
                print(" Connection failed (expected with test SSID)")
        except Exception as e:
            print(f" Connection error (expected): {str(e)[:100]}...")

        await client.disconnect()

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)

    # Test 2: Raw session ID format (for comparison)
    raw_session = "n1p5ah5u8t9438rbunpgrq0hlq"

    print("📝 Testing with raw session ID:")
    print(f"   Session: {raw_session}")
    print()

    try:
        # Create client with raw session
        client2 = AsyncPocketOptionClient(
            ssid=raw_session, is_demo=True, uid=72645361, platform=1
        )

        formatted_message2 = client2._format_session_message()

        print(" Client created successfully")
        print(f"📤 Formatted message: {formatted_message2[:50]}...")
        print(f"🔍 Session: {getattr(client2, 'session_id', 'N/A')}")
        print(f"👤 UID: {client2.uid}")
        print(f"🏷️  Platform: {client2.platform}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print(" SSID Format Tests Completed!")


async def test_real_connection():
    """Test with real SSID if available"""

    print("\nTesting Real Connection (Optional)")
    print("=" * 40)

    # Check for real SSID in environment
    real_ssid = os.getenv("POCKET_OPTION_SSID")

    if not real_ssid:
        print(" No real SSID found in environment variable POCKET_OPTION_SSID")
        print("   Set it like this for real testing:")
        print(
            '   export POCKET_OPTION_SSID=\'42["auth",{"session":"your_session","isDemo":1,"uid":your_uid,"platform":1}]\''
        )
        return

    print(f"🔑 Found real SSID: {real_ssid[:30]}...")

    try:
        client = AsyncPocketOptionClient(ssid=real_ssid)

        print("Attempting real connection...")
        await client.connect()

        if client.is_connected:
            print(" Successfully connected!")

            # Test basic functionality
            try:
                balance = await client.get_balance()
                print(f"Balance: ${balance.balance:.2f}")

                # Test health status
                health = await client.get_health_status()
                print(f"🏥 Health: {health}")

            except Exception as e:
                print(f" API error: {e}")

        else:
            print("Connection failed")

        await client.disconnect()
        print("Disconnected")

    except Exception as e:
        print(f"Connection error: {e}")


async def main():
    """Main test function"""

    print("PocketOption SSID Format Test Suite")
    print("=" * 60)
    print()

    # Test SSID format handling
    await test_complete_ssid_format()

    # Test real connection if available
    await test_real_connection()

    print("\n🎉 All tests completed!")
    print()
    print("📋 Usage Examples:")
    print("1. Complete SSID format (recommended):")
    print(
        '   ssid = r\'42["auth",{"session":"your_session","isDemo":1,"uid":your_uid,"platform":1}]\''
    )
    print("   client = AsyncPocketOptionClient(ssid=ssid)")
    print()
    print("2. Raw session format:")
    print(
        '   client = AsyncPocketOptionClient(ssid="your_session", uid=your_uid, is_demo=True)'
    )


if __name__ == "__main__":
    asyncio.run(main())
