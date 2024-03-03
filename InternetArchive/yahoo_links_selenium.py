import csv
import json
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import glob
import datetime
import pandas as pd

from multiprocessing import Pool, Manager, Lock
import traceback

num_process=10
opti = Options()
opti.set_preference("permissions.default.image", 2)
opti.set_preference("javascript.enabled", False)
opti.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)
opti.add_argument("-headless")  # Run in headless mode
serv = Service(executable_path='geckodriver')
# Set up Selenium WebDriver

char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0','-','_','$']
scraped = os.listdir('yahoo_links_1')
urls = []
for ch0 in char_list:
    for ch1 in char_list:
        if not f'yahoo_{ch0}{ch1}.txt' in scraped:
            urls.append(f'http://web.archive.org/cdx/search/?url=https://www.finance.yahoo.com/news/{ch0}{ch1}*')

print(urls)

def scrape_article_content(url,local_driver):
    filename = url.split('/')[-1].split('.html')[0]+ '.json'
    try:
        print(url)
        local_driver.get(url)
        WebDriverWait(local_driver, 3).until(
        lambda d: d.execute_script('return document.readyState') == 'complete')

        page_source = local_driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')

        text = soup.get_text(separator='\n', strip=True)
        print('#####################')
        filename = 'yahoo_'+url.split('news/')[-1].split('*')[0]+'.txt'
        with open('yahoo_links_1/'+filename, 'w') as file:
            file.write(text)
        
        input_filename = 'yahoo_links_1/'+filename
        output_filename = 'yahoo_links_1/'+'yahoo_'+url.split('news/')[-1].split('*')[0]+'.csv'

        df = pd.read_csv(input_filename, delimiter=' ', header=None, usecols=[1,2], names=['date_time','url'])

        print(df)
        # Filter rows where the URL contains '.html'
        df = df[df['url'].str.contains('.html')]

        # Truncate the URLs at '.html'
        df['url'] = df['url'].str.split('.html').str[0] + '.html'
        df['url'] = df['url'].str.replace(':80', '', regex=False)
        df['url'] = df['url'].str.replace('http:', 'https:', regex=False)

        # #Remove the specified pattern "%20 ... 2525252F"
        # pattern_to_remove = r'%20.*2F(?=.*2F)'
        # df['url'] = df['url'].apply(lambda x: re.sub(pattern_to_remove, '', x, flags=re.DOTALL | re.IGNORECASE))

        # Remove URLs containing "news/%"
        df = df[~df['url'].str.contains('news/%')]
        df = df[~df['url'].str.contains("news/'")]

        # Remove duplicates
        df.drop_duplicates(subset=['url'],inplace=True)
        print(df)
        # Write the unique URLs to a CSV file
        df.to_csv(output_filename, index=False)



    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def init_driver():
    service = Service(executable_path='geckodriver')
    driver = webdriver.Firefox(service=service, options=opti)
    return driver

def worker_task(url_list, lock):
    driver = init_driver()  # Initialize the driver for this worker
    while True:
        lock.acquire()
        try:
            if len(url_list) == 0:
                break  # If the list is empty, exit the loop
            url = url_list.pop()  # Pop the last URL from the shared list
        finally:
            lock.release()
        
        # Now that we have a URL, we can scrape it
        scrape_article_content(url, driver)
    
    driver.quit()  # Quit the driver when done    


if __name__ == "__main__":
    if(urls):
        manager = Manager()
        url_list = manager.list(urls)  # Shared list of URLs
        lock = manager.Lock()  # A lock to prevent simultaneous access to the list

        with Pool(processes=num_process) as pool:
            # Start the worker task for each process
            for _ in range(num_process):
                pool.apply_async(worker_task, args=(url_list, lock))

            pool.close()  # Close the pool to any new tasks
            pool.join()  # Wait for all worker processes to finish

# Define the directory where the CSV files are stored
folder_path = 'yahoo_links_1'

# Use glob to find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Initialize an empty list to store DataFrames
dfs = []

df = pd.read_csv('finance_yahoo_links.csv')

dfs.append(df)

# Loop over the list of csv files
for csv_file in csv_files:
    # Read the current CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Append the DataFrame to the list
    dfs.append(df)


# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)


merged_df.drop_duplicates(subset=['url'],inplace=True)
print(merged_df)
# (Optional) Save the merged DataFrame to a new CSV file
output_file = 'yahoo_links_1.csv'
merged_df.to_csv(output_file, index=False)

