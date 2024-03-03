import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# a=np.linspace(0x00,0xbb,num=5)
# print(a)
# for i in range(-10,5):
#     print(i)


# a = 0xee00ee
# b = 0xffffff
# hex_string = "#{:02x}{:02x}{:02x}".format(22,33,44)  # convert to hexadecimal string
# print(hex_string)


# fig, ax=plt.subplots()
# ax.plot([0, 1, 2], [0, 1, 2], color=hex_string)
# ax.tick_params(colors='white',which='both')
# ax.spines['bottom'].set_color('white')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# ax.spines['left'].set_color('white')
# ax.set_facecolor('black')
# ax.grid(c='#666666')
# fig.set_facecolor('black')
# plt.show()

import pandas as pd

df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']},
    index=[0, 1, 2, 3])

df2 = pd.DataFrame({
    'A': ['A4', 'A5', 'A6', 'A7'],
    'B': ['B4', 'B5', 'B6', 'B7'],
    'C': ['C4', 'C5', 'C6', 'C7'],
    'D': ['D4', 'D5', 'D6', 'D7']},
    index=[4,5,6,7])

df = pd.concat([df1,df2],axis=0)
print(df)

# Create an original DataFrame with a non-continuous DateTime index
idx = pd.DatetimeIndex(['2021-01-01', '2021-01-05', '2021-01-07'])
df = pd.DataFrame(range(len(idx)), index=idx, columns=['value'])

print("Original DataFrame:")
print(df)

# Identify the latest date in the index
latest_date = df.index.max()

# Generate 10 new dates starting from the day after the latest date
new_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=10)

# Create a new DataFrame with these dates, filled with zeros
new_df = pd.DataFrame(0, index=new_dates, columns=df.columns)

# Append the new DataFrame to the original DataFrame
df=pd.concat([df,new_df],axis=0)

print("\nDataFrame after appending new dates:")
print(df)