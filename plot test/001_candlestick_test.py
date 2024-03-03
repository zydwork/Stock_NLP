import pandas as pd
import matplotlib.pyplot as plt

prices=pd.read_csv("data\AAPL.csv")
prices['Date'] = pd.to_datetime(prices['Date'])
prices.set_index('Date',inplace=True)

#create figure
fig,ax=plt.subplots()

#define width of candlestick elements
width = 0.4
width2 = 0.05

#define up and down prices
up = prices[prices.Close>=prices.Open]
down = prices[prices.Close<prices.Open]

#define colors to use
col1 = 'green'
col2 = 'red'

a=1

#plot up prices
ax.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1,alpha=a)
ax.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1,alpha=a)
ax.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1,alpha=a)

#plot down prices
ax.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2,alpha=a)
ax.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2,alpha=a)
ax.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2,alpha=a)

ax.tick_params(colors='white',which='both')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.set_facecolor('black')
ax.grid(c='#666666')
fig.set_facecolor('black')

#rotate x-axis tick labels
# plt.xticks(rotation=45, ha='right')

# # Create a secondary y-axis
# ax2 = ax.twinx()

# # Plot the volume data on the secondary y-axis
# ax2.plot(prices.index, prices['Volume'], color='blue', linewidth=0.5)

# # Optionally adjust the y-axis label for clarity
# ax2.set_ylabel('Volume', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# #display candlestick chart
plt.show()