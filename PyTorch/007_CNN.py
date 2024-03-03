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

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(device)

pd.options.display.max_rows = 99
pd.options.display.max_columns = 999

lookback=-100 #must be a negative value
lookforward=30 #positive value
batchsize=64
start_date='2020-01-01'
end_date='2030-01-01'
ticker='AAPL'
LSTM_hidden_size=1
LSTM_num_stacked_layers=1
date=datetime.date.today()
time=datetime.datetime.now().strftime("%H-%M-%S")
num_epochs = 500
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
# print(data)

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

    df=dc(df)

    df.dropna(inplace=True)

    return df

shifted_df = dataframe_prep(data, lookback, lookforward, data_attributes)

# Identify the latest date in the index
data.set_index('Date',inplace=True)
latest_date = data.index.max()

print('latestdate',latest_date)

# Generate 10 new dates starting from the day after the latest date
new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=lookforward)

# Create a new DataFrame with these dates, filled with zeros
new_df = pd.DataFrame(170, index=new_dates, columns=data.columns)

# Append the new DataFrame to the original DataFrame
full_shifted_df=pd.concat([data,new_df],axis=0)

def full_dataframe_prep(origional_data, n_steps_backward, n_steps_forward, name):
    df = dc(origional_data)
    dataframes = [df]
    
    for name in data_attributes[:]:
        for i in range(n_steps_backward, 1):
            df_shifted=pd.DataFrame({})
            df_shifted[f'{name}(t{i})'] = df[f'{name}'].shift(-i)
            dataframes.append(df_shifted)
            # df[f'{name}(t{i})'] = df[f'{name}'].shift(-i)



    df = pd.concat(dataframes, axis=1)    
    #df.set_index('Date', inplace=True)

    df=dc(df)

    df.dropna(inplace=True)

    return df

full_shifted_df=full_dataframe_prep(data, lookback, lookforward, data_attributes)

# # Identify the latest date in the index
# latest_date = full_shifted_df.index.max()
# print('latestdate',latest_date)
# # Generate 10 new dates starting from the day after the latest date
# new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=lookforward)

# # Create a new DataFrame with these dates, filled with zeros
# new_df = pd.DataFrame(170, index=new_dates, columns=full_shifted_df.columns)

# # Append the new DataFrame to the original DataFrame
# full_shifted_df=pd.concat([full_shifted_df,new_df],axis=0)

print("\nDataFrame after appending new dates:")
print(full_shifted_df)
print(full_shifted_df.shape)

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

new_attri_dict={}

for i in range(lookback, 1):
    for name in data_attributes[:]:
        origional_nparr=full_shifted_df[f'{name}(t{i})'].to_numpy()
        new_attri_dict[f'{name}(t{i})']=np.reshape(origional_nparr,(-1,1,1))

#print(attri_dict)

temp_list=[ [[]] for _ in full_shifted_df.iterrows()]
temp=np.array(temp_list)



full_shifted_np_arr_list=[ [[ name for name in data_attributes[:] ]] for _ in full_shifted_df.iterrows()]
full_shifted_np_arr=np.array(full_shifted_np_arr_list)

for i in range(lookback,1):
    for name in data_attributes[:]:
        temp=np.concatenate((temp,new_attri_dict[f'{name}(t{i})']),axis=2)
    full_shifted_np_arr=np.concatenate((full_shifted_np_arr,temp),axis=1)
    temp=np.array(temp_list)

full_shifted_np_arr=np.delete(full_shifted_np_arr,0,axis=1)

#print(shifted_np_arr)
print(full_shifted_np_arr)

scaler = MinMaxScaler(feature_range=(-1, 1))
original_shape = shifted_np_arr.shape
shifted_np_arr = shifted_np_arr.reshape(-1, 1)
scaler.fit(shifted_np_arr)
shifted_np_arr = scaler.transform(shifted_np_arr)
shifted_np_arr = shifted_np_arr.reshape(original_shape)

original_shape = full_shifted_np_arr.shape
full_shifted_np_arr = full_shifted_np_arr.reshape(-1, 1)
full_shifted_np_arr = scaler.transform(full_shifted_np_arr)
full_shifted_np_arr = full_shifted_np_arr.reshape(original_shape)

print(shifted_np_arr)
print(full_shifted_np_arr)
print(shifted_np_arr.shape)
print(full_shifted_np_arr.shape)

x=shifted_np_arr[:,0:-lookback,:]
x_f=full_shifted_np_arr[:,0:-lookback,:]
#x=np.expand_dims(x,axis=2)
y=shifted_np_arr[:,-lookforward:,:]
#y=np.expand_dims(y,axis=1)

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=None,shuffle=False)
x_train=torch.tensor(x_train).float()
y_train=torch.tensor(y_train).float()
x_test=torch.tensor(x_test).float()
y_test=torch.tensor(y_test).float()
x_full=torch.tensor(x_f).float()
#y_full=torch.tensor(y).float()

#print(y_train.shape)
print('***************')

#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

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
#full_dataset=TimeSeriesDataset(x_full,y_full)

bs=batchsize #batch size

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
#full_loader = DataLoader(full_dataset, batch_size=bs, shuffle=False)

# for _, batch in enumerate(train_loader):
#     x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#     print(x_batch,y_batch)
#     print(x_batch.shape, y_batch.shape)
#     break

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers,output_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),            
            nn.Conv1d(in_channels=512, out_channels=input_size, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),            
        )
        #self.conv1 = nn.Conv1d(in_channels=len(data_attributes), out_channels=64, kernel_size=2)
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # self.lstm = nn.LSTM(output_size, hidden_size, num_stacked_layers,
                            # batch_first=True,dropout=0.2)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out = self.cnn(x)
        # out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -lookforward:, :])
        return out



model = CNN_LSTM(-lookback, LSTM_hidden_size, LSTM_num_stacked_layers, len(data_attributes))
model.to(device)
bestmodel=model
print(model)
#print(bestmodel)
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

best_loss=10
best_epoch=0

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
    global bestmodel,best_loss,best_epoch
    if avg_loss_across_batches<best_loss:
        bestmodel=model
        best_loss=avg_loss_across_batches
        best_epoch=epoch


    
    if epoch % disp==(disp-1):
        print('Val Loss:  {0:.9f}'.format(avg_loss_across_batches))
        print('Best Loss: {0:.9f}'.format(best_loss))
        print('Best epoch:',best_epoch)
        print('***************************************************')
        print()

learning_rate = 0.001

loss_function = nn.MSELoss()
l2_reg_strength=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg_strength)

for epoch in range(num_epochs):
    disp=10
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(x_full.to(device)).to('cpu').numpy()

print('xfull',x_full.shape)

model=bestmodel
torch.save(model.state_dict(), f"models\{ticker}_{date}_{time}.pth")

original_shape=predicted.shape
predicted=predicted.reshape(-1,1)
predicted=scaler.inverse_transform(predicted)
predicted=predicted.reshape(original_shape)



print(predicted)
print(predicted.shape)

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
new_drop_list=[]

for i in range(lookback,lookforward):
    for name in data_attributes:
        drop_list.append(f'{name}(t{i})')

for i in range(lookback,1):
    for name in data_attributes:
        new_drop_list.append(f'{name}(t{i})')
        

#print(drop_list)

full_shifted_df.drop(new_drop_list,axis=1,inplace=True)
full_shifted_df=dc(full_shifted_df)

print(full_shifted_df.shape)

for i,name in enumerate(data_attributes[:]):
    for j in range(lookforward):
        full_shifted_df[f'{name}(t+{j})']=predicted[:,j,i].flatten()
#shifted_df=output_to_pd(shifted_df,data_attributes,predicted,lookforward)

full_shifted_df=dc(full_shifted_df)

# Identify the latest date in the index
latest_date = full_shifted_df.index.max()
print('latestdate',latest_date)
# Generate 10 new dates starting from the day after the latest date
new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=lookforward)

# Create a new DataFrame with these dates, filled with zeros
new_df = pd.DataFrame(index=new_dates, columns=full_shifted_df.columns)

# Append the new DataFrame to the original DataFrame
full_shifted_df=pd.concat([full_shifted_df,new_df],axis=0)


for i,name in enumerate(data_attributes[:]):
    for j in range(lookforward):
        full_shifted_df[f'{name}(t+{j})']=full_shifted_df[f'{name}(t+{j})'].shift(j+1)

full_shifted_df=dc(full_shifted_df)
print(full_shifted_df)
full_shifted_df.to_csv('debug/final.csv')


fig, ax=plt.subplots()

#define width of candlestick elements
width = 0.4
width2 = 0.05

#define up and down shifted_df


dec_col=np.linspace(0,240,lookforward+1)

for lf in [5,15,29]: #range(lookforward):
    col=int(dec_col[lf])
    #col=0
    alp=0.5
    #lf=lookforward-1
    
    cola="#{:02x}{:02x}{:02x}".format(col,col,255)
    colb="#{:02x}{:02x}{:02x}".format(255,255,col)

    up1 = full_shifted_df[full_shifted_df[f'Close(t+{lf})']>=full_shifted_df[f'Open(t+{lf})']]
    down1 = full_shifted_df[full_shifted_df[f'Close(t+{lf})']<full_shifted_df[f'Open(t+{lf})']]
    ax.bar(up1.index,up1[f'Close(t+{lf})']-up1[f'Open(t+{lf})'],width,bottom=up1[f'Open(t+{lf})'],color=cola,alpha=alp)
    ax.bar(up1.index,up1[f'High(t+{lf})']-up1[f'Close(t+{lf})'],width2,bottom=up1[f'Close(t+{lf})'],color=cola,alpha=alp)
    ax.bar(up1.index,up1[f'Low(t+{lf})']-up1[f'Open(t+{lf})'],width2,bottom=up1[f'Open(t+{lf})'],color=cola,alpha=alp)

    ax.bar(down1.index,down1[f'Close(t+{lf})']-down1[f'Open(t+{lf})'],width,bottom=down1[f'Open(t+{lf})'],color=colb,alpha=alp)
    ax.bar(down1.index,down1[f'High(t+{lf})']-down1[f'Open(t+{lf})'],width2,bottom=down1[f'Open(t+{lf})'],color=colb,alpha=alp)
    ax.bar(down1.index,down1[f'Low(t+{lf})']-down1[f'Close(t+{lf})'],width2,bottom=down1[f'Close(t+{lf})'],color=colb,alpha=alp)
#define colors to use
col1 = 'green'
col2 = 'red'

up = full_shifted_df[full_shifted_df.Close>=full_shifted_df.Open]
down = full_shifted_df[full_shifted_df.Close<full_shifted_df.Open]
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