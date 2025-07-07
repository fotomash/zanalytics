# 🔧 MetaTrader 5 Connection Setup

## 1. Locate Your MT5 EA File
Use one of these (Simple version is recommended):
- `HttpsJsonSender_Simple.mq5` (Recommended)
- `HttpsJsonSender.mq5`

## 2. Update the EA Configuration
Open the MQ5 file and find these lines:

```mql5
input string   InpURL = "http://localhost:5000/webhook";  // API endpoint
input int      InpTimer = 1;                               // Timer in seconds
```

Make sure the URL matches your API server.

## 3. Install the EA
1. Copy the .mq5 file to: `<MT5 Data Folder>/MQL5/Experts/`
2. Open MetaEditor (F4 in MT5)
3. Compile the EA (F7)
4. Restart MT5

## 4. Attach the EA to a Chart
1. Open any chart (e.g., EURUSD)
2. Drag the EA from Navigator to the chart
3. In settings:
   - Enable "Allow algorithmic trading"
   - Enable "Allow DLL imports" (if required)
   - Verify the URL is correct
4. Click OK

## 5. Verify Connection
Check the Experts tab for messages like:
- "Sending data to server..."
- "Response: 200"

If you see error 4014, check:
- Is the API server running?
- Is the URL correct?
- Firewall settings

## 6. API Endpoints
Your API should have these endpoints:
- `POST /webhook` - Receives tick data
- `GET /status` - Health check
- `GET /data` - Retrieve processed data
