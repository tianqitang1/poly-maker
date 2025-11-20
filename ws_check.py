import asyncio, json
import websockets

# Tokens for the condition provided
TOKENS = [
    "31786876708530202714896016471227648729713105836189748042827693364109109385575",  # token1 (YES)
    "8499786491926678022568708652627152799277578595650304749692503778588993969205",   # token2 (NO)
]

async def dump_once():
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as ws:
        await ws.send(json.dumps({"assets_ids": TOKENS}))
        print("Subscribed")
        for i in range(5):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"\nMessage {i}:")
            if isinstance(data, dict) and data.get("event_type") == "book":
                print(f"asset_id: {data.get('asset_id')}")
                print(f"bids (top 3): {data.get('bids', [])[:3]}")
                print(f"asks (top 3): {data.get('asks', [])[:3]}")
            else:
                print(json.dumps(data, indent=2)[:1000])

if __name__ == "__main__":
    asyncio.run(dump_once())
