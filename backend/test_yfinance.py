import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print(f"yfinance version: {yf.__version__}")

# Test with a single symbol and recent dates
symbols = ['SPY', 'AAPL', 'MSFT']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"\nTesting download from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

try:
    # Test 1: Download all at once
    print("\nTest 1: Download all symbols together...")
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=True,
        group_by='ticker'
    )
    
    if not data.empty:
        print("✅ Success! Data shape:", data.shape)
        if isinstance(data, pd.DataFrame):
            print("Columns:", data.columns.tolist())
    else:
        print("❌ No data returned")

    # Test 2: Download individually
    print("\nTest 2: Download symbols individually...")
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if not hist.empty:
            print(f"✅ {symbol}: {len(hist)} days")
        else:
            print(f"❌ {symbol}: No data")
            
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check if symbols exist
print("\nTest 3: Check symbol info...")
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info
        if info and 'regularMarketPrice' in info:
            print(f"✅ {symbol}: Current price ${info['regularMarketPrice']}")
        else:
            print(f"⚠️ {symbol}: Limited info available")
    except:
        print(f"❌ {symbol}: Could not fetch info")
        