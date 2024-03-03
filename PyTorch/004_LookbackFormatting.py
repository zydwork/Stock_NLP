import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from copy import deepcopy as dc

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

pd.options.display.max_rows = 99
pd.options.display.max_columns = 999

data=pd.read_csv("AAPL.csv")
data_attributes=['Date','Open','High','Low','Close','Adj Close']
data=data[data_attributes]


print(data)


data['Date'] = pd.to_datetime(data['Date'])


def prepare_dataframe_for_lstm(origional_data, n_steps, name):
    df = dc(origional_data)
  
    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'{name}(t-{i})'] = df[f'{name}'].shift(i)

    df.dropna(inplace=True)

    return df

def dataframe_prep(origional_data,n_steps,name):
    df = dc(origional_data)
  
    for name in data_attributes[1:]:
        for i in range(0, n_steps+1):
            df[f'{name}(t-{i})'] = df[f'{name}'].shift(i)
        
    df.set_index('Date', inplace=True)

    df.dropna(inplace=True)

    return df

lookback = 3
shifted_df = dataframe_prep(data, lookback,data_attributes)
print(shifted_df)


scaler = MinMaxScaler(feature_range=(-1, 1))


attri_dict={}

for i in range(lookback,-1,-1):
    for name in data_attributes[1:]:
        origional_nparr=shifted_df[f'{name}(t-{i})'].to_numpy()
        attri_dict[f'{name}(t-{i})']=np.reshape(origional_nparr,(-1,1,1))
print(attri_dict)

# shifted_np_arr=np.array([[[]]])
# for i in shifted_df.iterrows():
#     print(i)
#     np.concatenate((shifted_np_arr,np.array([[[]]])),axis=1)

# print(shifted_np_arr.shape)
# temp=np.array([[[]]])

temp_list=[ [[]] for _ in shifted_df.iterrows()]
temp=np.array(temp_list)

shifted_np_arr_list=[ [[ None for _ in data_attributes[1:] ]] for _ in shifted_df.iterrows()]
shifted_np_arr=np.array(shifted_np_arr_list)

print(shifted_np_arr)
print(shifted_np_arr.shape)

#shifted_np_arr=dc(temp_list)

#temp=dc(temp_list)



# for name in data_attributes[1:]:
#     temp=np.append(temp,attri_dict[f'{name}(t-0)'],axis=2)
# print(temp)
# print(temp.shape)
# print(shifted_np_arr.shape)

for i in range(lookback,-1,-1):
    for name in data_attributes[1:]:
        temp=np.concatenate((temp,attri_dict[f'{name}(t-{i})']),axis=2)
    shifted_np_arr=np.concatenate((shifted_np_arr,temp),axis=1)
    temp=np.array(temp_list)
         

shifted_np_arr=np.delete(shifted_np_arr,0,axis=1)
print(shifted_np_arr)
print(shifted_np_arr.shape)

# np.concatenate




# for attribute in data_attributes[1:]:
#     attribute_dict[f'{attribute}']=\
#         prepare_dataframe_for_lstm(data[['Date',f'{attribute}']], lookback, attribute)
#     attribute_dict_as_np[f'{attribute}']=\
#         attribute_dict[f'{attribute}'].to_numpy()
#