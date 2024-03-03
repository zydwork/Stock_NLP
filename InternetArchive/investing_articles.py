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

num_process = 10
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
df = pd.read_csv('investing_links_1.csv')  # Replace with your CSV file path
#print(len(os.listdir('investing_articles')+os.listdir('investing_articles_empty')))
print(df)
successed=os.listdir('investing_articles')
failed=os.listdir('investing_articles_empty')
if successed:
    scraped_filenames = pd.DataFrame(successed+failed)
    scraped_filenames = scraped_filenames[0].str.split('.json').str[0]
    df['basename'] = df['url'].apply(
        lambda x: x.split('news/')[-1].split('.html')[0]
    )
    
    df = df[~df['basename'].isin(scraped_filenames)]
    df.drop(columns=['basename'], inplace=True)
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
        content = soup.find('div', class_='WYSIWYG articlePage')
        if content:
        # Find all <div> elements with the class 'relatedInstrumentsWrapper' within the article div
            for related in content.find_all('div', class_='relatedInstrumentsWrapper'):
        # Decompose removes the tag from the tree and destroys it along with its contents
                related.decompose()

        title = ''
        title = soup.find('h1', class_="articleHeader").get_text()
        article_text = content.get_text(separator='\n', strip=True)

        content_sections = soup.find_all('div', class_='contentSectionDetails')

        source = ''
        date_time = ''
        date_time_updated = ''


        for section in content_sections:
            # Check if the 'style' attribute is present in the <div>
            if section.has_attr('style'):
                # This is the first <div>; find the <img> tag and get the 'src' attribute
                img_tag = section.find('img')
                if img_tag and img_tag.has_attr('src'):
                    source = img_tag['src']
                    source = source.split("/")[-1].split(".png")[0]
                # This is the second <div>; find all <span> elements without any child tags

            else:
                spans = [span for span in section.find_all('span') if not span.find()]

                # The first <span> is always the date of the article
                date_time = spans[0].get_text(strip=True).split('Published ')[-1] if spans else ''

                # If there's a second <span>, it's the updated date of the article
                if len(spans) > 1:
                    date_time_updated = spans[1].get_text(strip=True).split('Updated ')[-1]


        
        
        # local_driver.quit()
        # Construct the JSON object
        data = {
            'title': title,
            'url': url,
            'source': source,
            'source_url': '',
            'date_time': date_time,
            'date_time_updated': date_time_updated,
            'article_text': article_text,
        }
        #print(data)

        # Write to a JSON file
        if title:
            with open('investing_articles/'+filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print("FILE SAVED  "+filename)
        else:
            with open('investing_articles_empty/'+filename, 'w') as f:
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
        with open('investing_articles_empty/'+filename, 'w') as f:
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

