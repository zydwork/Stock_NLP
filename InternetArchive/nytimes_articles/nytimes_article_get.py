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

start_date = datetime.date(2015, 1, 1)
end_date = datetime.date.today()

# Set up Selenium WebDriver
serv = Service(executable_path='geckodriver')
opti = Options()
opti.set_preference("permissions.default.image", 2)
opti.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)
opti.headless = True  # Run in headless mode
driver = webdriver.Firefox(service=serv, options=opti)

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
        article_body = soup#.find('section', name_='articleBody')
        if not article_body:
            print(f"Couldn't find the article body div for URL: {url}")
            return None

        # Scrape the title and article text
        title = soup.find('title').text if soup.find('title') else 'No Title Found'
        article_text = '\n'.join(p.text for p in article_body.find_all('p'))
        print(article_text)

        return {
            'link': url,
            'title': title,
            'article_text': article_text
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


current_date = start_date
# Read URLs from the CSV file
while current_date <= end_date:

    article_links = []
    date_string = current_date.strftime('%Y_%m_%d')
    csv_file=f'nytimes_links/nytimes_{date_string}.csv'
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row
        for row in csvreader:
            article_links.append(row[0])

    # Scrape articles
    articles_data = []
    for link in article_links:
        article_data = scrape_article_content(link)
        print(article_data)
        if article_data:
            articles_data.append(article_data)

    # Write data to JSON file
    with open(f'nytimes_articles/nytimes_article_{date_string}.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(articles_data, jsonfile, ensure_ascii=False, indent=4)
        
    current_date += datetime.timedelta(days=1)

# Close the Selenium WebDriver
driver.quit()