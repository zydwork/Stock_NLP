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
import random

num_process=8
opti = Options()
opti.set_preference("permissions.default.image", 2)
opti.set_preference("javascript.enabled", False)
opti.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)
opti.add_argument("-headless")  # Run in headless mode
serv = Service(executable_path='geckodriver')
# Set up Selenium WebDriver


# driver_dict = {i: webdriver.Firefox(service=Service(executable_path='geckodriver'), options=opti) for i in range(num_process)}
# driver_list = [i for i in range(num_process)]

# for i in range(num_process):
#     service_dict[i] = Service(executable_path='geckodriver')
#     driver_dict[i] = webdriver.Firefox(service=service_dict[i], options=opti)

# print(driver_dict)
# print(driver_list)

# Read the CSV file to get the URLs
df = pd.read_csv('yahoo_links_a.csv')  # Replace with your CSV file path
#print(len(os.listdir('yahoo_articles')+os.listdir('yahoo_articles_empty')))
scraped_filenames = pd.DataFrame(os.listdir('yahoo_articles')+os.listdir('yahoo_articles_empty')+os.listdir('yahoo_articles_failed_1'))
scraped_filenames = scraped_filenames[0].str.split('.json').str[0]
df['basename'] = df['url'].apply(
    lambda x: x.split('news/')[-1].split('.html')[0]
)
print(df)
df = df[~df['basename'].isin(scraped_filenames)]
df.drop(columns=['basename'], inplace=True)
df = df[~df['url'].str.contains("\?")]
# Filter URLs to only include subdomains behind "news/" that start with the specified letters
allowed_subdomains = ['ca','cw','ya','y-']
df['subdomain'] = df['url'].apply(lambda x: x.split('news/')[1][0:2] if 'news/' in x else '')
df = df[df['subdomain'].str.lower().isin(allowed_subdomains)]
df.drop(columns=['subdomain'], inplace=True)
print(df)
urls = df['url'].tolist()
print(len(urls))

def scrape_article_content(url,local_driver):
    filename = url.split('/')[-1].split('.html')[0]+ '.json'
    try:
        local_driver.get(url)
        WebDriverWait(local_driver, 3).until(
        lambda d: d.execute_script('return document.readyState') == 'complete')

        page_source = local_driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')
        #print(soup)
        # Check for 'mainArticle' div, otherwise parse the whole document
        main_article = soup.find('div', id='mainArticle')
        content = main_article if main_article else soup
        text = soup.get_text(separator='\n', strip=True)
        article = content.find('div', class_='caas-body')
        title = content.find('title').get_text() if content.find('title') else ''
        article_text = article.get_text(separator='\n', strip=True)

        caas_logo = content.find('div', class_='caas-logo')

        source = caas_logo.find('span', class_='caas-attr-provider').get_text() if caas_logo else ''
        source_url = caas_logo.find('a')['href'] if caas_logo and caas_logo.find('a') else ''
        date_time = content.find('time')['datetime']

        
        
        # local_driver.quit()
        # Construct the JSON object
        data = {
            'title': title,
            'url': url,
            'source': source,
            'source_url': source_url,
            'date_time': date_time,
            'article_text': article_text,
        }
        #print(data)

        # Write to a JSON file
        if title:
            with open('yahoo_articles/'+filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("FILE SAVED  "+filename)
        else:
            with open('yahoo_articles_failed_1/'+filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


    except Exception as e:
        print(f"Error scraping {url}: {e}")
        data = {
            'title': '',
            'url': url,
            'source': '',
            'source_url': '',
            'date_time': '',
            'article_text': '',
        }
        with open('yahoo_articles_failed_1/'+filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        #local_driver.quit()
        return None

def init_driver():
    service = Service(executable_path='geckodriver')
    driver = webdriver.Firefox(service=service, options=opti)
    return driver

def worker_task(url_list, lock):
    driver = init_driver()  # Initialize the driver for this worker
    while True:
        lock.acquire()
        list_length = len(url_list)
        try:
            if list_length == 0:
                break  # If the list is empty, exit the loop
            random_index = random.randint(0, list_length - 1)
            url = url_list.pop(random_index)  # Pop the last URL from the shared list
        finally:
            lock.release()
        
        # Now that we have a URL, we can scrape it
        scrape_article_content(url, driver)
    
    driver.quit()  # Quit the driver when done    


if __name__ == "__main__":
    df = pd.read_csv('finance_yahoo_links.csv')
    manager = Manager()
    url_list = manager.list(urls)  # Shared list of URLs
    lock = manager.Lock()  # A lock to prevent simultaneous access to the list

    with Pool(processes=num_process) as pool:
        # Start the worker task for each process
        for _ in range(num_process):
            pool.apply_async(worker_task, args=(url_list, lock))

        pool.close()  # Close the pool to any new tasks
        pool.join()  # Wait for all worker processes to finish

# # Iterate over the URLs from the CSV and scrape data
# if __name__ == "__main__":        
#     with Pool(processes=5) as pool:
#         # Map the URLs to the worker processes
#         pool.map(scrape_article_content, urls)

# for driver in driver_dict.values():
#     driver.quit
# # Close the WebDriver

