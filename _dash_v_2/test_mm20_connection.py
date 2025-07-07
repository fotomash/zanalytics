import requests

# Test connection to mm20.local
try:
    response = requests.get("http://mm20.local:8080/symbols")
    if response.status_code == 200:
        symbols = response.json()
        print(f"✅ Connected! Found {len(symbols)} symbols:")
        for symbol in symbols[:5]:
            print(f"  - {symbol}")
    else:
        print(f"❌ Error: {response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect: {e}")
    print("Make sure:")
    print("1. API is running on mm20.local")
    print("2. Port 8080 is open")
    print("3. You can ping mm20.local")
