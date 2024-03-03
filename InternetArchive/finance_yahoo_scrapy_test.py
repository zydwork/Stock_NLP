import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
from bs4 import BeautifulSoup
import json
import os

# Define the Spider
class YahooFinanceSpider(scrapy.Spider):
    name = 'yahoofinance'

    def start_requests(self):
        # Read the CSV file with pandas to get the URLs
        # df = pd.read_csv('finance_yahoo_links.csv')  # Make sure the path to your CSV file is correct
        urls = ['https://finance.yahoo.com/news/1-biden-harris-target-republican-180521904.html']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Use BeautifulSoup to parse the HTML content
        print('\n\n\n$$$$$\n\n\n')
        soup = BeautifulSoup(response.text, 'html.parser')
        print("\n\n\n\n##########\n\n\n")
        #print(soup)
        paragraphs = soup.find_all('p')

        # Extract data with BeautifulSoup
        text = soup.get_text(separator='  ', strip=True)
        print(text)
        print('#####################')

        # Check for 'mainArticle' div, otherwise parse the whole document
        main_article = soup.find('div', id='mainArticle')
        content = main_article if main_article else soup

        # Extract the required information
        title = content.find('title').get_text() if content.find('title') else ''
        
        # The article text
        article_text = ' '.join(p.text for p in content.find_all('p'))

        # The source and source URL
        caas_logo = content.find('div', class_='caas-logo')
        source = caas_logo.find('span', class_='caas-attr-provider').get_text() if caas_logo else ''
        source_url = caas_logo.find('a')['href'] if caas_logo and caas_logo.find('a') else ''

        # Construct the filename from URL
        filename = response.url.split('/')[-1].split('.html')[0]+ '.json'
        
        # Construct the JSON object
        data = {
            'title': title,
            'url': response.url,
            'article_text': article_text,
            'source': source,
            'source_url': source_url
        }
        print(data)
        # Write to a JSON file
        # with open('yahoo_articles/'+filename, 'w') as f:
        #     json.dump(data, f)

# Main script to run the spider
if __name__ == "__main__":
    # Define a process with the Spider
    process = CrawlerProcess()
    process.crawl(YahooFinanceSpider)
    process.start()
    
    # Confirm completion
    print("Scraping completed.")