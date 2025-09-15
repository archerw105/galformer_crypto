import yfinance as yf
import pandas as pd
import datetime

# Today's date
end = datetime.datetime.now()

# 300 days ago
days= 3000
start = end - datetime.timedelta(days=days)

# Download daily data for BTC-USD
df = yf.download(
    tickers="BTC-USD",
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    interval="1d"
)

# Reset index to make Date a column
df.reset_index(inplace=True)

# Flatten column names (remove multi-level index)
df.columns = [col[0] if col[1] == 'BTC-USD' else col[0] for col in df.columns]

# Add Adj Close column (same as Close for crypto)
df['Adj Close'] = df['Close']

# Reorder columns to match desired format
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Save to CSV
df.to_csv(f"Datasets/btc_daily_last_{days}_days.csv", index=False)
print(df.head())
