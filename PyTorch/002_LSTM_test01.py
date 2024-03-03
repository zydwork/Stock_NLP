#https://medium.com/@mrconnor/time-series-forecasting-with-pytorch-predicting-stock-prices-81db0f4348ef

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

pd.options.display.max_rows = 99
pd.options.display.max_columns = 999

data=pd.read_csv("AAPL.csv")
# print(data)
data=data[['Date','Close']]
# print(data)

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

data['Date'] = pd.to_datetime(data['Date'])

# print(data['Date'])
# print(type(data['Date']))

#plt.plot(data['Date'], data['Close'])
#plt.pause(0.001)

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 20
shifted_df = prepare_dataframe_for_lstm(data, lookback)
#print(shifted_df)

shifted_df_as_np = shifted_df.to_numpy()
print(shifted_df_as_np)

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

print(shifted_df_as_np.shape)
print(shifted_df_as_np)

x=shifted_df_as_np[:,1:]
print(x.shape)
x=np.expand_dims(x,axis=2)
print(x.shape)
y=shifted_df_as_np[:,0]
y=np.expand_dims(y,axis=1)

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=None,shuffle=False)
x_train=torch.tensor(x_train).float()
y_train=torch.tensor(y_train).float()
x_test=torch.tensor(x_test).float()
y_test=torch.tensor(y_test).float()

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

print(y_train)
bs=16 #batch size

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# for _, batch in enumerate(train_loader):
#     x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#     print(x_batch,y_batch)
#     print(x_batch.shape, y_batch.shape)
#     break

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
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
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    disp=10
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(x_train.to(device)).to('cpu').numpy()

print(y_train)
print(predicted)

train_predictions = predicted.flatten()
y_compare=y_train.flatten()

print(train_predictions.shape)
print(x_train.shape)
print(x_train.shape[0])
dummies = np.zeros((x_train.shape[0], lookback+1))
print(dummies.shape)
dummies[:, 0] = train_predictions
print(dummies.shape)
dummies = scaler.inverse_transform(dummies)
print(dummies)

dummies_y = np.zeros((y_train.shape[0], lookback+1))
dummies_y[:, 0] = y_compare
dummies_y = scaler.inverse_transform(dummies_y)

print(y_compare)

train_predictions = dc(dummies[:, 0])
y_compare=dc(dummies_y[:,0])

plt.plot(y_compare, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

