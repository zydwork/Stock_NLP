#https://medium.com/@mrconnor/time-series-forecasting-with-pytorch-predicting-stock-shifted_df-81db0f4348ef

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
import pickle
import datetime


pd.options.display.max_rows = 99
pd.options.display.max_columns = 999

lookback=-60 #must be a negative value
lookforward=10 #positive value
batchsize=32
start_date='2010-01-01'
end_date='2030-01-01'
ticker='AMZN'
LSTM_hidden_size=6
LSTM_num_stacked_layers=3
date=datetime.date.today()
time=datetime.datetime.now().strftime("%H-%M-%S")
num_epochs = 200
print(time)
data=pd.read_csv(f"csv\sp500\{ticker}.csv")
data_attributes=['Date','Open','High','Low','Close','Adj Close','Volume']
data=data[data_attributes]
data_attributes.remove('Date')

vol_arr=data['Volume'].to_numpy()
data['Volume']=vol_arr/10000000

data['Date'] = pd.to_datetime(data['Date'])
mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
data=data.loc[mask]
print(data)

def dataframe_prep(origional_data, n_steps_backward, n_steps_forward, name):
    df = dc(origional_data)
    dataframes = [df]
    
    for name in data_attributes[:]:
        for i in range(n_steps_backward, n_steps_forward):
            df_shifted=pd.DataFrame({})
            df_shifted[f'{name}(t{i})'] = df[f'{name}'].shift(-i)
            dataframes.append(df_shifted)
            # df[f'{name}(t{i})'] = df[f'{name}'].shift(-i)

    df = pd.concat(dataframes, axis=1)    
    df.set_index('Date', inplace=True)

    df.dropna(inplace=True)

    return df

shifted_df = dataframe_prep(data, lookback, 1, data_attributes)

#print(shifted_df)

attri_dict={}

for i in range(lookback, lookforward):
    for name in data_attributes[:]:
        origional_nparr=shifted_df[f'{name}(t{i})'].to_numpy()
        attri_dict[f'{name}(t{i})']=np.reshape(origional_nparr,(-1,1,1))

#print(attri_dict)

temp_list=[ [[]] for _ in shifted_df.iterrows()]
temp=np.array(temp_list)

shifted_np_arr_list=[ [[ name for name in data_attributes[:] ]] for _ in shifted_df.iterrows()]
shifted_np_arr=np.array(shifted_np_arr_list)

for i in range(lookback,lookforward):
    for name in data_attributes[:]:
        temp=np.concatenate((temp,attri_dict[f'{name}(t{i})']),axis=2)
    shifted_np_arr=np.concatenate((shifted_np_arr,temp),axis=1)
    temp=np.array(temp_list)

shifted_np_arr=np.delete(shifted_np_arr,0,axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
original_shape = shifted_np_arr.shape
shifted_np_arr = shifted_np_arr.reshape(-1, 1)
shifted_np_arr = scaler.fit_transform(shifted_np_arr)
shifted_np_arr = shifted_np_arr.reshape(original_shape)

#print(shifted_np_arr)
#print(shifted_np_arr.shape)

x=shifted_np_arr[:,0:-lookback,:]
#x=np.expand_dims(x,axis=2)
y=shifted_np_arr[:,-lookforward:,:]
#y=np.expand_dims(y,axis=1)

x_full=torch.tensor(x).float()

print('***************')


    


for epoch in range(num_epochs):
    disp=10
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(x_full.to(device)).to('cpu').numpy()

model=bestmodel
torch.save(model.state_dict(), f"models\{ticker}_{date}_{time}.pth")

original_shape=predicted.shape
predicted=predicted.reshape(-1,1)
predicted=scaler.inverse_transform(predicted)
predicted=predicted.reshape(original_shape)



#print(predicted)
#print(predicted.shape)

#print(shifted_df[data_attributes[:]])

def output_to_pd(shifted_data,attri,pred,lf):
    df = dc(shifted_data)
    dataframes=[]
    #pd.DataFrame(shifted_data.index,columns=['Date'])
    for i,name in enumerate(attri[:]):
        df_name = shifted_data[[f'{name}']].reset_index()
        dataframes.append(df_name)
        for j in range(lf):
            #print(i,name)
            df_shifted=pd.DataFrame({})
            df_shifted[f'{name}(t+{j})'] = pred[:,j,i].flatten()
            dataframes.append(df_shifted)
            #shifted_df[f'{name}(t+{j})']=
    df=pd.concat(dataframes,axis=1)
    print(df)
    df.set_index('Date', inplace=True)
    return df

drop_list=[]

for i in range(lookback,lookforward):
    for name in data_attributes:
        drop_list.append(f'{name}(t{i})')
        

#print(drop_list)

shifted_df.drop(drop_list,axis=1,inplace=True)
shifted_df=dc(shifted_df)


for i,name in enumerate(data_attributes[:]):
    for j in range(lookforward):
        shifted_df[f'{name}(t+{j})']=predicted[:,j,i].flatten()
#shifted_df=output_to_pd(shifted_df,data_attributes,predicted,lookforward)

shifted_df=dc(shifted_df)

#print(shifted_df)


fig, ax=plt.subplots()

#define width of candlestick elements
width = 0.4
width2 = 0.05

#define up and down shifted_df


dec_col=np.linspace(0,220,lookforward+1)

for lf in range(1):
    lf=lookforward-1
    col=0
    alp=0.5
    
    cola="#{:02x}{:02x}{:02x}".format(col,col,255)
    colb="#{:02x}{:02x}{:02x}".format(255,255,col)

    up1 = shifted_df[shifted_df[f'Close(t+{lf})']>=shifted_df[f'Open(t+{lf})']]
    down1 = shifted_df[shifted_df[f'Close(t+{lf})']<shifted_df[f'Open(t+{lf})']]
    ax.bar(up1.index,up1[f'Close(t+{lf})']-up1[f'Open(t+{lf})'],width,bottom=up1[f'Open(t+{lf})'],color=cola,alpha=alp)
    ax.bar(up1.index,up1[f'High(t+{lf})']-up1[f'Close(t+{lf})'],width2,bottom=up1[f'Close(t+{lf})'],color=cola,alpha=alp)
    ax.bar(up1.index,up1[f'Low(t+{lf})']-up1[f'Open(t+{lf})'],width2,bottom=up1[f'Open(t+{lf})'],color=cola,alpha=alp)

    ax.bar(down1.index,down1[f'Close(t+{lf})']-down1[f'Open(t+{lf})'],width,bottom=down1[f'Open(t+{lf})'],color=colb,alpha=alp)
    ax.bar(down1.index,down1[f'High(t+{lf})']-down1[f'Open(t+{lf})'],width2,bottom=down1[f'Open(t+{lf})'],color=colb,alpha=alp)
    ax.bar(down1.index,down1[f'Low(t+{lf})']-down1[f'Close(t+{lf})'],width2,bottom=down1[f'Close(t+{lf})'],color=colb,alpha=alp)
#define colors to use
col1 = 'green'
col2 = 'red'

up = shifted_df[shifted_df.Close>=shifted_df.Open]
down = shifted_df[shifted_df.Close<shifted_df.Open]
#plot up shifted_df
ax.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
ax.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
ax.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

#plot down shifted_df
ax.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
ax.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
ax.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

ax.tick_params(colors='white',which='both')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.set_facecolor('black')
ax.grid(c='#666666')
fig.set_facecolor('black')

plt.xticks(rotation=45, ha='right')

plt.show()
# with open(f"plots\{ticker}_plot_{date}_{time}.pkl", 'wb') as file:
#     pickle.dump(ax, file)
print(f"plots\{ticker}_plot_{date}_{time}.pkl")


log_data={
        'datetime':datetime.datetime.now(),
        'lookback':lookback,
        'lookforward':lookforward,
        'batchsize':batchsize,
        'startdate':end_date,
        'ticker':ticker,
        'LSTM_hidden_size':LSTM_hidden_size,
        'LSTM_num_stacked_layers':LSTM_num_stacked_layers,
        'epochs':num_epochs,
        'bestloss':best_loss,
        'bestepoch':best_epoch,
        'modelfile':f"models\{ticker}_model_{date}_{time}.pth",
        'plotfile':f"plots\{ticker}_plot_{date}_{time}.pkl",
}

# create a binary pickle file  


# write the python object (dict) to pickle file
pickle.dump(log_data, open(f"log_data\{ticker}.pkl","wb"))

# close file
# # Create a secondary y-axis
# ax2 = plt.twinx()

# # Plot the volume data on the secondary y-axis
# ax2.plot(shifted_df.index, shifted_df['Volume'], color='blue', linewidth=0.5)

# # Optionally adjust the y-axis label for clarity
# ax2.set_ylabel('Volume', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

#display candlestick chart