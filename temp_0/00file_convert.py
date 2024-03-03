import pandas as pd
import os

# data=pd.read_parquet('240M_Nor240/20190104/150000000.pqt')
# print(data)
# data.to_csv('sample.csv')
direc='240M_Nor240'
folders = os.listdir(direc)
folders = [folder for folder in folders \
           if os.path.isdir(direc+'/'+folder) and folder.isdigit() and len(folder) == 8]

#tickers = 
print(folders)

