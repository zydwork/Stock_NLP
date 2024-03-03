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

start_date = datetime.date(2015, 1, 1)
end_date = datetime.date.today()

# Set up Selenium WebDriver
serv = Service(executable_path='geckodriver')
opti = Options()
opti.set_preference("permissions.default.image", 2)
opti.set_preference("javascript.enabled", False)
opti.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)
opti.headless = True  # Run in headless mode
driver = webdriver.Firefox(service=serv, options=opti)


# Read the CSV file to get the URLs
df = pd.read_csv('finance_yahoo_links.csv')  # Replace with your CSV file path
urls = df['url'].tolist()

def scrape_article_content(url):
    try:
        # Navigate to the article page
        driver.get(url)

        # Wait for the page to load
        WebDriverWait(driver, 3).until(
        lambda d: d.execute_script('return document.readyState') == 'complete'
        )

        # Scrape page source and close the driver
        page_source = driver.page_source

        # Initialize BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        #print(soup)
        # Check for 'mainArticle' div, otherwise parse the whole document
        main_article = soup.find('div', id='mainArticle')
        # print(main_article)
        content = main_article if main_article else soup
        # print(content)
        text = soup.get_text(separator='\n', strip=True)
        #print(text)
        article = content.find('div', class_='caas-body')
        #print(article)
        # Extract the required information
        title = content.find('title').get_text() if content.find('title') else ''
        #print(title)
        # The article text
        article_text = article.get_text(separator='\n', strip=True)
        #article_text = '\n'.join(p.text for p in article.find_all('span'))
        #print(article_text)
        # The source and source URL
        caas_logo = content.find('div', class_='caas-logo')
        #print(caas_logo)
        source = caas_logo.find('span', class_='caas-attr-provider').get_text() if caas_logo else ''
        #print(source)
        source_url = caas_logo.find('a')['href'] if caas_logo and caas_logo.find('a') else ''
        #print(source_url)
        date_time = content.find('time')['datetime']
        #print(date_time)
        # Construct the filename from URL
        filename = url.split('/')[-1].split('.html')[0]+ '.json'
        
        # Construct the JSON object
        data = {
            'title': title,
            'url': url,
            'source': source,
            'source_url': source_url,
            'date_time': date_time,
            'article_text': article_text,


        }
        print(data)

        # Write to a JSON file
        if title:
            with open('yahoo_articles/'+filename, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Iterate over the URLs from the CSV and scrape data
for url in urls:
    scrape_article_content(url)
        

# Close the WebDriver
driver.quit()
