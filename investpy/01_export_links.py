import pandas as pd
import investpy as inv

# Load the ticker list from the CSV file
ticker_list = pd.read_csv('HSI.csv')

# Add a new column for the links
ticker_list['link'] = ''

# Iterate through each row in the dataframe
for index, row in ticker_list.iterrows():
    try:
        # Try to get the company profile using investpy
        symbol=row['symbol'].split('.')[0]
        print(index,symbol)
        info = inv.get_stock_company_profile(stock=symbol, country="hong kong")
        # If the profile is retrieved successfully, add the URL to the dataframe
        link = info['url'].replace('-company-profile','')
        #print(link)
        ticker_list.at[index, 'link'] = link
    except Exception as e:
        # If an error occurs (e.g., the profile is not found), leave the cell blank
        print(f"An error occurred for ticker {row['symbol']}: {e}")
        continue

# Print the dataframe with the links
print(ticker_list)

# Save the updated dataframe to a new CSV file
ticker_list.to_csv('HSI_with_links.csv', index=False)
ticker_list['link'].to_csv('HSI_only_link.csv',index=False)