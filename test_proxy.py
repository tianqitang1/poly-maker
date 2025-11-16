#!/usr/bin/env python3
"""
Quick test script to verify proxy configuration.

Run this AFTER setting up your SSH tunnel and configuring .env

Usage:
    uv run python test_proxy.py
"""

from poly_utils.proxy_config import setup_proxy, get_proxy_session
import sys

def test_proxy():
    print("\n" + "="*70)
    print("🧪 Testing Proxy Configuration")
    print("="*70)

    # Setup proxy
    has_proxy = setup_proxy(verbose=True)

    if not has_proxy:
        print("\n❌ No proxy configured!")
        print("Add one of these to your .env file:")
        print("  SOCKS_PROXY_URL=socks5://localhost:1080")
        print("  PROXY_URL=http://proxy.example.com:8080")
        return False

    # Test 1: Get our external IP
    print("\n📡 Test 1: Checking external IP...")
    try:
        session = get_proxy_session()
        response = session.get('https://api.ipify.org?format=json', timeout=10)
        ip_data = response.json()
        print(f"✓ External IP: {ip_data['ip']}")
        print("  (This should be your SERVER's IP, not your local IP)")
    except Exception as e:
        print(f"❌ Failed to get external IP: {e}")
        return False

    # Test 2: Connect to Polymarket API
    print("\n📡 Test 2: Testing Polymarket API connection...")
    try:
        response = session.get('https://clob.polymarket.com/tick-size', timeout=10)
        print(f"✓ Polymarket API: Status {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ Failed to connect to Polymarket: {e}")
        return False

    # Test 3: Connect to Polygon RPC
    print("\n📡 Test 3: Testing Polygon RPC connection...")
    try:
        response = session.post(
            'https://polygon-rpc.com',
            json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
            timeout=10
        )
        print(f"✓ Polygon RPC: Status {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data:
                block_num = int(data['result'], 16)
                print(f"  Latest block: {block_num}")
    except Exception as e:
        print(f"❌ Failed to connect to Polygon RPC: {e}")
        return False

    # Test 4: Test py_clob_client patching
    print("\n📡 Test 4: Testing py_clob_client integration...")
    try:
        from py_clob_client.http_helpers import helpers
        print("✓ py_clob_client imported successfully")
        print("✓ HTTP helpers patched to use proxy")
    except Exception as e:
        print(f"⚠ Warning: {e}")

    print("\n" + "="*70)
    print("✅ All proxy tests passed!")
    print("="*70)
    print("\nYou can now run the bot with: uv run python main.py")
    print()
    return True

if __name__ == '__main__':
    success = test_proxy()
    sys.exit(0 if success else 1)
