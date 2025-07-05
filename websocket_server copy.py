# websocket_server.py
   import asyncio
   import websockets
   import json
   import aiohttp
   
   async def forward_to_api(data):
       """Forward WebSocket data to your FastAPI endpoint"""
       async with aiohttp.ClientSession() as session:
           # Convert tick/bar data to candle format if needed
           if "tick" in data:
               # Transform tick data to candle format
               candle_data = {
                   "candle": {
                       "timestamp": data["tick"]["time"],
                       "symbol": data["tick"]["symbol"],
                       "timeframe": "M1",  # Default to 1-minute
                       "open": data["tick"]["bid"],
                       "high": data["tick"]["bid"],
                       "low": data["tick"]["bid"],
                       "close": data["tick"]["bid"],
                       "volume": 1
                   }
               }
           elif "bar" in data:
               # Bar data is already in OHLC format
               candle_data = {
                   "candle": {
                       "timestamp": data["bar"]["time"],
                       "symbol": data["bar"]["symbol"],
                       "timeframe": data["bar"]["period"],
                       "open": data["bar"]["open"],
                       "high": data["bar"]["high"],
                       "low": data["bar"]["low"],
                       "close": data["bar"]["close"],
                       "volume": data["bar"]["volume"]
                   }
               }
           
           # Send to your API
           async with session.post("http://localhost:8000/api/v1/ingest/candle", 
                                  json=candle_data) as response:
               return await response.json()
   
   async def handle_websocket(websocket, path):
       """Handle incoming WebSocket connections"""
       print(f"Client connected: {websocket.remote_address}")
       try:
           async for message in websocket:
               print(f"Received: {message}")
               try:
                   data = json.loads(message)
                   result = await forward_to_api(data)
                   print(f"API result: {result}")
                   # Send back acknowledgment
                   await websocket.send(json.dumps({"status": "processed", "result": result}))
               except json.JSONDecodeError:
                   print("Invalid JSON received")
                   await websocket.send(json.dumps({"error": "Invalid JSON"}))
               except Exception as e:
                   print(f"Error processing message: {e}")
                   await websocket.send(json.dumps({"error": str(e)}))
       except websockets.exceptions.ConnectionClosed:
           print("Client disconnected")
   
   async def main():
       server = await websockets.serve(handle_websocket, "localhost", 8765)
       print("WebSocket server started at ws://localhost:8765")
       await server.wait_closed()
   
   if __name__ == "__main__":
       asyncio.run(main())