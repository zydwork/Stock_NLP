import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = "cpu" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

# Download historical data
#data = yf.download('AAPL', start='2010-01-01', end='2023-10-01', interval='1d')

#Read data from downloaded csv file
data=pd.read_csv("AAPL.csv")

# Preprocessing
data = data['Close'].values
print(type(data),data.shape)
data = data.reshape(-1, 1)
print(data)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Create windows
def create_windows(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


X, y = create_windows(data, 60)

print(X.shape())
print(X)
print(y.shape())
print(y)

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.layer1=nn.Linear(input_dim,hidden_dim)
        self.activation1=nn.ReLU()
        self.layer2=nn.Linear(hidden_dim,hidden_dim)
        self.activation2=nn.ReLU()
        self.layer3=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=self.activation1(self.layer1(x))
        x=self.activation2(self.layer2(x))
        x=self.layer3(x)
        return x
    
model=MLP(input_dim=2,hidden_dim=64,output_dim=1).to(device)

# Reshape for CNN and convert to PyTorch tensors
X = X.reshape((X.shape[0], X.shape[1], 1))
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# # Create CNN model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(60, 32, 5)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(56*32, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         return x

# model = CNN()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print('Test loss:', loss.item())