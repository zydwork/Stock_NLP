import yahooquery as yq
ticker = yq.Ticker('lin').price['lin']['marketCap']
print(ticker)