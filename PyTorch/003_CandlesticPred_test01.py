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

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

pd.options.display.max_rows = 99
pd.options.display.max_columns = 999

data=pd.read_csv("csv\AAPL_1.csv")
data_attributes=['Date','Open','High','Low','Close','Adj Close','Volume']
data=data[data_attributes]

vol_arr=data['Volume'].to_numpy()
data['Volume']=vol_arr/10000000
print(data['Volume'])

data['Date'] = pd.to_datetime(data['Date'])

def dataframe_prep(origional_data,n_steps,name):
    df = dc(origional_data)
  
    for name in data_attributes[1:]:
        for i in range(0, n_steps+1):
            df[f'{name}(t-{i})'] = df[f'{name}'].shift(i)
        
    df.set_index('Date', inplace=True)

    df.dropna(inplace=True)

    return df

lookback = 10
shifted_df = dataframe_prep(data, lookback,data_attributes)


attri_dict={}

for i in range(lookback,-1,-1):
    for name in data_attributes[1:]:
        origional_nparr=shifted_df[f'{name}(t-{i})'].to_numpy()
        attri_dict[f'{name}(t-{i})']=np.reshape(origional_nparr,(-1,1,1))

temp_list=[ [[]] for _ in shifted_df.iterrows()]
temp=np.array(temp_list)

shifted_np_arr_list=[ [[ None for _ in data_attributes[1:] ]] for _ in shifted_df.iterrows()]
shifted_np_arr=np.array(shifted_np_arr_list)

for i in range(lookback,-1,-1):
    for name in data_attributes[1:]:
        temp=np.concatenate((temp,attri_dict[f'{name}(t-{i})']),axis=2)
    shifted_np_arr=np.concatenate((shifted_np_arr,temp),axis=1)
    temp=np.array(temp_list)

shifted_np_arr=np.delete(shifted_np_arr,0,axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
original_shape = shifted_np_arr.shape
shifted_np_arr = shifted_np_arr.reshape(-1, 1)
shifted_np_arr = scaler.fit_transform(shifted_np_arr)
shifted_np_arr = shifted_np_arr.reshape(original_shape)

print(shifted_np_arr.shape)
print(shifted_np_arr)

x=shifted_np_arr[:,1:,:]
#x=np.expand_dims(x,axis=2)
y=shifted_np_arr[:,0,:]
#y=np.expand_dims(y,axis=1)

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=None,shuffle=False)
x_train=torch.tensor(x_train).float()
y_train=torch.tensor(y_train).float()
x_test=torch.tensor(x_test).float()
y_test=torch.tensor(y_test).float()
x_full=torch.tensor(x).float()
y_full=torch.tensor(y).float()

print(y_train)
print('***************')

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        return self.x[i],self.y[i]
    
train_dataset=TimeSeriesDataset(x_train,y_train)
test_dataset=TimeSeriesDataset(x_test,y_test)
full_dataset=TimeSeriesDataset(x_full,y_full)

print(y_train)
bs=32 #batch size

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
full_loader = DataLoader(full_dataset, batch_size=bs, shuffle=False)

# for _, batch in enumerate(train_loader):
#     x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#     print(x_batch,y_batch)
#     print(x_batch.shape, y_batch.shape)
#     break

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(6, 8, 4, 6)
model.to(device)
print(model)

def train_one_epoch():
    model.train(True)
    if epoch % disp==(disp-1):
        print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % disp==(disp-1):
            if batch_index % 50 == 49:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.9f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    if epoch % disp==(disp-1):
        print('Val Loss: {0:.9f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

learning_rate = 0.001
num_epochs = 200
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    disp=10
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(x_full.to(device)).to('cpu').numpy()

torch.save(model.state_dict(), "model.pth")

original_shape=predicted.shape
predicted=predicted.reshape(-1,1)
predicted=scaler.inverse_transform(predicted)
predicted=predicted.reshape(original_shape)


print(predicted)
print(predicted.shape)

print(shifted_df[data_attributes[1:]])

for i,name in enumerate(data_attributes[1:]):
    print(i,name)
    shifted_df[f'{name}(t+1)']=predicted[:,i].flatten()

print(shifted_df)

plt.figure()

#define width of candlestick elements
width = 0.4
width2 = 0.05

#define up and down shifted_df
up = shifted_df[shifted_df.Close>=shifted_df.Open]
down = shifted_df[shifted_df.Close<shifted_df.Open]

lookforward_n=1

up1 = shifted_df[shifted_df[f'Close(t+{lookforward_n})']>=shifted_df[f'Open(t+{lookforward_n})']]
down1 = shifted_df[shifted_df[f'Close(t+{lookforward_n})']<shifted_df[f'Open(t+{lookforward_n})']]

#define colors to use
col1 = 'green'
col2 = 'red'
col3 = 'blue'
col4 = 'orange'

#plot up shifted_df
plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

#plot down shifted_df
plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

plt.bar(up1.index,up1[f'Close(t+{lookforward_n})']-up1[f'Open(t+{lookforward_n})'],width,bottom=up1[f'Open(t+{lookforward_n})'],color=col3)
plt.bar(up1.index,up1[f'High(t+{lookforward_n})']-up1[f'Close(t+{lookforward_n})'],width2,bottom=up1[f'Close(t+{lookforward_n})'],color=col3)
plt.bar(up1.index,up1[f'Low(t+{lookforward_n})']-up1[f'Open(t+{lookforward_n})'],width2,bottom=up1[f'Open(t+{lookforward_n})'],color=col3)

plt.bar(down1.index,down1[f'Close(t+{lookforward_n})']-down1[f'Open(t+{lookforward_n})'],width,bottom=down1[f'Open(t+{lookforward_n})'],color=col4)
plt.bar(down1.index,down1[f'High(t+{lookforward_n})']-down1[f'Open(t+{lookforward_n})'],width2,bottom=down1[f'Open(t+{lookforward_n})'],color=col4)
plt.bar(down1.index,down1[f'Low(t+{lookforward_n})']-down1[f'Close(t+{lookforward_n})'],width2,bottom=down1[f'Close(t+{lookforward_n})'],color=col4)

plt.xticks(rotation=45, ha='right')

plt.show()

# # Create a secondary y-axis
# ax2 = plt.twinx()

# # Plot the volume data on the secondary y-axis
# ax2.plot(shifted_df.index, shifted_df['Volume'], color='blue', linewidth=0.5)

# # Optionally adjust the y-axis label for clarity
# ax2.set_ylabel('Volume', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

#display candlestick chart