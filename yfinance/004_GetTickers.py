import pandas as pd
import numpy as np
import yfinance as yf

sp500=pd.read_csv('csv\lists\sp500list.csv')
sp500_Symb=sp500['Symbol'].to_list()
print(sp500_Symb)

for name in sp500_Symb:
    data = yf.download(f'{name}', start='1900-01-01', end='2030-01-01', interval='1d')
    data.to_csv(f'csv\sp500\{name}.csv')

