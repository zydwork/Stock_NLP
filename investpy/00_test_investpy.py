import pandas as pd
import investpy as inv

ticker_list=pd.read_csv('Nikkei.csv')
print(ticker_list)

TICKER = "4523"

# get company's profile
info=inv.get_stock_company_profile(stock=TICKER, country="japan")
print(info['url'])