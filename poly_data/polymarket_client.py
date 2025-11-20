from dotenv import load_dotenv          # Environment variable management
import os                           # Operating system interface

# Polymarket API client libraries
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType, PartialCreateOrderOptions
from py_clob_client.constants import POLYGON

# Web3 libraries for blockchain interaction
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

import requests                     # HTTP requests
import pandas as pd                 # Data analysis
import json                         # JSON processing
import subprocess                   # For calling external processes

from py_clob_client.clob_types import OpenOrderParams

# Smart contract ABIs
from poly_data.abis import NegRiskAdapterABI, ConditionalTokenABI, erc20_abi
from poly_utils.proxy_config import (
    get_web3_provider_with_proxy,
    setup_proxy,
)

# Load environment variables
load_dotenv()


class PolymarketClient:
    """
    Client for interacting with Polymarket's API and smart contracts.
    
    This class provides methods for:
    - Creating and managing orders
    - Querying order book data
    - Checking balances and positions
    - Merging positions
    
    The client connects to both the Polymarket API and the Polygon blockchain.
    """
    
    def __init__(self, pk='default', use_near_sure=False, account_type='default') -> None:
        """
        Initialize the Polymarket client with API and blockchain connections.

        Args:
            pk (str, optional): Legacy parameter, kept for backward compatibility
            use_near_sure (bool, optional): Legacy parameter, use account_type instead
            account_type (str): Account type - 'default', 'near_sure', or 'neg_risk_arb'
        """
        host="https://clob.polymarket.com"

        # Ensure py_clob_client HTTP traffic also uses the configured proxy (needed to bypass Cloudflare blocks)
        setup_proxy(verbose=False)

        # Handle legacy use_near_sure parameter for backward compatibility
        if use_near_sure and account_type == 'default':
            account_type = 'near_sure'

        # Get credentials based on account type
        if account_type == 'near_sure':
            key = os.getenv("NEAR_SURE_PK")
            browser_address = os.getenv("NEAR_SURE_BROWSER_ADDRESS")
            print("Initializing Polymarket client for NEAR-SURE account...")
            env_prefix = "NEAR_SURE_"
        elif account_type == 'neg_risk_arb':
            key = os.getenv("NEG_RISK_ARB_PK")
            browser_address = os.getenv("NEG_RISK_ARB_BROWSER_ADDRESS")
            print("Initializing Polymarket client for NEG-RISK-ARB account...")
            env_prefix = "NEG_RISK_ARB_"
        elif account_type == 'sports_arb':
            key = os.getenv("SPORTS_ARB_PK")
            browser_address = os.getenv("SPORTS_ARB_BROWSER_ADDRESS")
            print("Initializing Polymarket client for SPORTS-ARB account...")
            env_prefix = "SPORTS_ARB_"
        elif account_type == 'og_maker':
            key = os.getenv("OG_MAKER_PK")
            browser_address = os.getenv("OG_MAKER_BROWSER_ADDRESS")
            print("Initializing Polymarket client for OG-MAKER account...")
            env_prefix = "OG_MAKER_"
        else:  # default
            key = os.getenv("PK")
            browser_address = os.getenv("BROWSER_ADDRESS")
            print("Initializing Polymarket client...")
            env_prefix = ""

        if not key or not browser_address:
            raise ValueError(
                f"Missing environment variables: {env_prefix}PK and/or {env_prefix}BROWSER_ADDRESS. "
                f"Please set them in your .env file."
            )

        chain_id=POLYGON
        self.browser_wallet=Web3.to_checksum_address(browser_address)

        # Initialize the Polymarket API client
        # Note: Use signature_type=2 for Gnosis Safe wallets, 1 for Magic wallets
        self.client = ClobClient(
            host=host,
            key=key,
            chain_id=chain_id,
            funder=self.browser_wallet,
            signature_type=2  # 2 for Gnosis Safe/browser wallets, 1 for email/Magic wallet
        )

        # Set up API credentials
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds=self.creds)

        # Initialize Web3 connection to Polygon with proxy support
        provider = get_web3_provider_with_proxy("https://polygon-rpc.com")
        web3 = Web3(provider)
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        # Set up USDC contract for balance checks
        self.usdc_contract = web3.eth.contract(
            address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", 
            abi=erc20_abi
        )

        # Store key contract addresses
        self.addresses = {
            'neg_risk_adapter': '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296',
            'collateral': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
            'conditional_tokens': '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045'
        }

        # Initialize contract interfaces
        self.neg_risk_adapter = web3.eth.contract(
            address=self.addresses['neg_risk_adapter'], 
            abi=NegRiskAdapterABI
        )

        self.conditional_tokens = web3.eth.contract(
            address=self.addresses['conditional_tokens'], 
            abi=ConditionalTokenABI
        )

        self.web3 = web3

    
    def create_order(self, marketId, action, price, size, neg_risk=False, raise_on_error=False):
        """
        Create and submit a new order to the Polymarket order book.
        
        Args:
            marketId (str): ID of the market token to trade
            action (str): "BUY" or "SELL"
            price (float): Order price (0-1 range for prediction markets)
            size (float): Order size in USDC
            neg_risk (bool, optional): Whether this is a negative risk market. Defaults to False.
            raise_on_error (bool, optional): If True, raises exceptions instead of printing and returning empty dict. Defaults to False.
            
        Returns:
            dict: Response from the API containing order details, or empty dict on error
        """
        # Create order parameters
        order_args = OrderArgs(
            token_id=str(marketId),
            price=price,
            size=size,
            side=action
        )

        signed_order = None

        # Handle regular vs negative risk markets differently
        if neg_risk == False:
            signed_order = self.client.create_order(order_args)
        else:
            signed_order = self.client.create_order(order_args, options=PartialCreateOrderOptions(neg_risk=True))
            
        try:
            # Submit the signed order to the API
            resp = self.client.post_order(signed_order)  # OrderType.GTC is default
            return resp
        except Exception as ex:
            if raise_on_error:
                raise ex
            print(ex)
            return {}

    def get_market(self, condition_id):
        """
        Get market information by condition ID.
        
        Args:
            condition_id (str): Market condition ID
            
        Returns:
            dict: Market information
        """
        try:
            return self.client.get_market(condition_id)
        except Exception as e:
            print(f"Error fetching market {condition_id}: {e}")
            return None

    def get_order_book(self, market):
        """
        Get the current order book for a specific market.

        Args:
            market (str): Market ID to query

        Returns:
            tuple: (bids_df, asks_df) - DataFrames containing bid and ask orders
        """
        try:
            orderBook = self.client.get_order_book(market)

            # Validate order book structure
            if not hasattr(orderBook, 'bids') or not hasattr(orderBook, 'asks'):
                print(f"Warning: Order book for {market} missing bids or asks")
                return pd.DataFrame(), pd.DataFrame()

            bids = orderBook.bids if orderBook.bids is not None else []
            asks = orderBook.asks if orderBook.asks is not None else []

            bids_df = pd.DataFrame(bids).astype(float) if bids else pd.DataFrame()
            asks_df = pd.DataFrame(asks).astype(float) if asks else pd.DataFrame()

            return bids_df, asks_df

        except Exception as e:
            print(f"Error fetching order book for {market}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_order_book_dict(self, market):
        """
        Get order book as simple dict structure compatible with strategies that
        expect price/size lists instead of DataFrames.

        Returns:
            dict: {'bids': [{'price': float, 'size': float}, ...], 'asks': [...]}
        """
        try:
            order_book = self.client.get_order_book(market)
        except Exception as e:
            print(f"Error fetching order book for {market}: {e}")
            return {}

        bids = getattr(order_book, 'bids', None) if not isinstance(order_book, dict) else order_book.get('bids', [])
        asks = getattr(order_book, 'asks', None) if not isinstance(order_book, dict) else order_book.get('asks', [])

        def _normalize(side):
            normalized = []
            if not side:
                return normalized

            for entry in side:
                if isinstance(entry, dict):
                    price = float(entry.get('price', 0))
                    size = float(entry.get('size', entry.get('amount', 0)))
                    normalized.append({'price': price, 'size': size})
                elif hasattr(entry, 'price'):
                    price = float(getattr(entry, 'price', 0))
                    size = float(getattr(entry, 'size', getattr(entry, 'amount', 0)))
                    normalized.append({'price': price, 'size': size})
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    # Assume [price, size, ...]
                    normalized.append({'price': float(entry[0]), 'size': float(entry[1])})
            return normalized

        bids_norm = sorted(_normalize(bids), key=lambda x: x['price'], reverse=True)
        asks_norm = sorted(_normalize(asks), key=lambda x: x['price'])

        return {'bids': bids_norm, 'asks': asks_norm}


    def get_usdc_balance(self):
        """
        Get the USDC balance of the connected wallet.
        
        Returns:
            float: USDC balance in decimal format
        """
        return self.usdc_contract.functions.balanceOf(self.browser_wallet).call() / 10**6
     
    def get_pos_balance(self):
        """
        Get the total value of all positions for the connected wallet.

        Returns:
            float: Total position value in USDC
        """
        from poly_utils.proxy_config import get_proxy_session
        session = get_proxy_session()
        res = session.get(f'https://data-api.polymarket.com/value?user={self.browser_wallet}')
        return float(res.json()['value'])

    def get_total_balance(self):
        """
        Get the combined value of USDC balance and all positions.
        
        Returns:
            float: Total account value in USDC
        """
        return self.get_usdc_balance() + self.get_pos_balance()

    def get_all_positions(self):
        """
        Get all positions for the connected wallet across all markets.

        Returns:
            DataFrame: All positions with details like market, size, avgPrice
        """
        from poly_utils.proxy_config import get_proxy_session
        session = get_proxy_session()
        res = session.get(f'https://data-api.polymarket.com/positions?user={self.browser_wallet}')
        return pd.DataFrame(res.json())
    
    def get_raw_position(self, tokenId):
        """
        Get the raw token balance for a specific market outcome token.
        
        Args:
            tokenId (int): Token ID to query
            
        Returns:
            int: Raw token amount (before decimal conversion)
        """
        return int(self.conditional_tokens.functions.balanceOf(self.browser_wallet, int(tokenId)).call())

    def get_position(self, tokenId):
        """
        Get both raw and formatted position size for a token.
        
        Args:
            tokenId (int): Token ID to query
            
        Returns:
            tuple: (raw_position, shares) - Raw token amount and decimal shares
                   Shares less than 1 are treated as 0 to avoid dust amounts
        """
        raw_position = self.get_raw_position(tokenId)
        shares = float(raw_position / 1e6)

        # Ignore very small positions (dust)
        if shares < 1:
            shares = 0

        return raw_position, shares
    
    def get_all_orders(self):
        """
        Get all open orders for the connected wallet.
        
        Returns:
            DataFrame: All open orders with their details
        """
        orders_df = pd.DataFrame(self.client.get_orders())

        # Convert numeric columns to float
        for col in ['original_size', 'size_matched', 'price']:
            if col in orders_df.columns:
                orders_df[col] = orders_df[col].astype(float)

        return orders_df
    
    def get_market_orders(self, market):
        """
        Get all open orders for a specific market.
        
        Args:
            market (str): Market ID to query
            
        Returns:
            DataFrame: Open orders for the specified market
        """
        orders_df = pd.DataFrame(self.client.get_orders(OpenOrderParams(
            market=market,
        )))

        # Convert numeric columns to float
        for col in ['original_size', 'size_matched', 'price']:
            if col in orders_df.columns:
                orders_df[col] = orders_df[col].astype(float)

        return orders_df
    

    def cancel_all_asset(self, asset_id):
        """
        Cancel all orders for a specific asset token.
        
        Args:
            asset_id (str): Asset token ID
        """
        self.client.cancel_market_orders(asset_id=str(asset_id))


    
    def cancel_all_market(self, marketId):
        """
        Cancel all orders in a specific market.
        
        Args:
            marketId (str): Market ID
        """
        self.client.cancel_market_orders(market=marketId)

    
    def merge_positions(self, amount_to_merge, condition_id, is_neg_risk_market):
        """
        Merge positions in a market to recover collateral.
        
        This function calls the external poly_merger Node.js script to execute
        the merge operation on-chain. When you hold both YES and NO positions
        in the same market, merging them recovers your USDC.
        
        Args:
            amount_to_merge (int): Raw token amount to merge (before decimal conversion)
            condition_id (str): Market condition ID
            is_neg_risk_market (bool): Whether this is a negative risk market
            
        Returns:
            str: Transaction hash or output from the merge script
            
        Raises:
            Exception: If the merge operation fails
        """
        amount_to_merge_str = str(amount_to_merge)

        # Prepare the command to run the JavaScript script
        node_command = f'node poly_merger/merge.js {amount_to_merge_str} {condition_id} {"true" if is_neg_risk_market else "false"}'
        print(node_command)

        # Run the command and capture the output
        result = subprocess.run(node_command, shell=True, capture_output=True, text=True)
        
        # Check if there was an error
        if result.returncode != 0:
            print("Error:", result.stderr)
            raise Exception(f"Error in merging positions: {result.stderr}")
        
        print("Done merging")

        # Return the transaction hash or output
        return result.stdout
