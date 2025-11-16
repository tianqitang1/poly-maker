"""
Proxy Configuration for Poly-Maker Bot

This module configures proxy support for all HTTP/HTTPS requests made by the bot,
including:
- Polymarket API calls (via py_clob_client)
- Polygon RPC calls (via Web3)
- Direct requests calls

Usage:
    from poly_utils.proxy_config import setup_proxy
    setup_proxy()  # Reads from environment variables
"""

import os
import requests
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Global session with proxy configuration
_proxy_session = None


def get_proxy_config() -> Optional[Dict[str, str]]:
    """
    Get proxy configuration from environment variables.

    Environment variables:
        PROXY_URL: Full proxy URL (e.g., "http://proxy.example.com:8080")
        SOCKS_PROXY_URL: SOCKS proxy URL (e.g., "socks5://localhost:1080")
        HTTP_PROXY: HTTP proxy (alternative to PROXY_URL)
        HTTPS_PROXY: HTTPS proxy (alternative to PROXY_URL)
        NO_PROXY: Comma-separated list of hosts to bypass proxy

    Returns:
        Dict with 'http' and 'https' proxy URLs, or None if no proxy configured
    """
    # Check for SOCKS proxy first (most common for SSH tunnels)
    socks_proxy = os.getenv('SOCKS_PROXY_URL')
    if socks_proxy:
        logger.info(f"Using SOCKS proxy: {socks_proxy}")
        return {
            'http': socks_proxy,
            'https': socks_proxy
        }

    # Check for single PROXY_URL
    proxy_url = os.getenv('PROXY_URL')
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        return {
            'http': proxy_url,
            'https': proxy_url
        }

    # Check for separate HTTP/HTTPS proxies
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')

    if http_proxy or https_proxy:
        config = {}
        if http_proxy:
            config['http'] = http_proxy
            logger.info(f"Using HTTP proxy: {http_proxy}")
        if https_proxy:
            config['https'] = https_proxy
            logger.info(f"Using HTTPS proxy: {https_proxy}")
        return config

    return None


def create_proxied_session() -> requests.Session:
    """
    Create a requests Session with proxy configuration.

    Returns:
        Configured requests.Session object
    """
    session = requests.Session()

    proxy_config = get_proxy_config()
    if proxy_config:
        session.proxies.update(proxy_config)
        logger.info("Proxy configuration applied to session")
    else:
        logger.info("No proxy configuration found, using direct connection")

    # Set timeout for all requests
    session.timeout = 30

    # Add retry logic for better reliability
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def get_proxy_session() -> requests.Session:
    """
    Get the global proxy-configured session (singleton pattern).

    Returns:
        Shared requests.Session with proxy configuration
    """
    global _proxy_session
    if _proxy_session is None:
        _proxy_session = create_proxied_session()
    return _proxy_session


def patch_requests_for_clob_client():
    """
    Monkey-patch requests in py_clob_client to use our proxied session.

    This ensures all HTTP calls made by py_clob_client go through the proxy.
    """
    try:
        from py_clob_client.http_helpers import helpers

        # Store original request function
        original_request = requests.request
        session = get_proxy_session()

        # Create wrapper that uses our session
        def proxied_request(method, url, **kwargs):
            # Use session's request method instead of module-level function
            return session.request(method, url, **kwargs)

        # Replace the request function in helpers module
        helpers.requests.request = proxied_request

        logger.info("✓ Patched py_clob_client to use proxy")
        return True

    except ImportError:
        logger.warning("py_clob_client not found, skipping patch")
        return False
    except Exception as e:
        logger.error(f"Failed to patch py_clob_client: {e}")
        return False


def get_web3_provider_with_proxy(rpc_url: str = "https://polygon-rpc.com"):
    """
    Create a Web3 HTTPProvider with proxy support.

    Args:
        rpc_url: Polygon RPC endpoint URL

    Returns:
        HTTPProvider configured with proxy
    """
    from web3 import Web3
    from web3.providers import HTTPProvider

    proxy_config = get_proxy_config()

    if proxy_config:
        # Web3.py HTTPProvider uses requests under the hood
        # We can pass a custom session with proxy configuration
        session = get_proxy_session()

        # Create provider with custom session
        provider = HTTPProvider(
            rpc_url,
            request_kwargs={'timeout': 30}
        )

        # Inject our proxied session
        provider.make_request = lambda method, params: session.post(
            rpc_url,
            json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
            timeout=30
        ).json()

        logger.info(f"✓ Created Web3 provider with proxy: {rpc_url}")
        return provider
    else:
        # No proxy, use default
        return HTTPProvider(rpc_url)


def setup_proxy(verbose: bool = True):
    """
    Setup proxy configuration for the entire bot.

    This function should be called ONCE at bot startup, before initializing
    any API clients.

    Args:
        verbose: If True, print proxy configuration status

    Returns:
        True if proxy is configured, False otherwise
    """
    if verbose:
        print("\n" + "="*60)
        print("🔧 Configuring Proxy Support")
        print("="*60)

    proxy_config = get_proxy_config()

    if proxy_config:
        if verbose:
            print(f"✓ Proxy detected:")
            for key, value in proxy_config.items():
                print(f"  {key.upper()}: {value}")

        # Create the global session
        get_proxy_session()

        # Patch py_clob_client
        patch_requests_for_clob_client()

        if verbose:
            print("✓ All HTTP traffic will be routed through proxy")
            print("="*60 + "\n")

        return True
    else:
        if verbose:
            print("ℹ No proxy configuration found")
            print("  To use a proxy, set one of these environment variables:")
            print("    PROXY_URL=http://proxy.example.com:8080")
            print("    SOCKS_PROXY_URL=socks5://localhost:1080")
            print("    HTTP_PROXY / HTTPS_PROXY")
            print("="*60 + "\n")

        return False


# Auto-setup when module is imported (optional)
# Uncomment if you want automatic proxy setup on import
# setup_proxy(verbose=False)
