import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import os
import pandas as pd

string = ''
class MySpider(scrapy.Spider):
    name = 'myspider'
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0',
        'FEEDS': {
            'investing_links_new_raw.txt': {
                'format': 'txt',
                'store_empty': False,
            },
        },
        'DOWNLOAD_WARNSIZE':0,
        # 'DOWNLOAD_DELAY': 0.2,
        # 'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
    }

    def start_requests(self):
        # Define the list of URLs to scrape here
        char_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0','-','_','$']
        scraped = os.listdir('investing_links_1')
        for ch0 in char_list:
            for ch1 in char_list:
                if not f'investing_{ch0}{ch1}.txt' in scraped:
                    url = f'http://web.archive.org/cdx/search/?url=https://www.investing.com/news/{ch0}{ch1}*'
                    yield scrapy.Request(url=url, callback=self.parse)
        

    def parse(self, response):
        body = response.body.decode('utf-8')
        # Use BeautifulSoup to parse the response
        soup = BeautifulSoup(body, 'html.parser')
        # paragraphs = soup.find_all('p')

        # Extract data with BeautifulSoup
        text = soup.get_text(separator='\n', strip=True)
        #print(text)
        print('#####################')
        filename = 'investing_'+response.url.split('news/')[-1].split('*')[0]+'.txt'
        with open('investing_links_1/'+filename, 'w') as file:
            file.write(text)
        
        input_filename = 'investing_links_1/'+filename
        output_filename = 'investing_links_1/'+'investing_'+response.url.split('news/')[-1].split('*')[0]+'.csv'

        # Read the file into a Pandas DataFrame
        # Assuming the data doesn't contain any quoted delimiters, which would require setting `quoting=csv.QUOTE_NONE`
        df = pd.read_csv(input_filename, delimiter=' ', header=None, usecols=[1,2], names=['date_time','url'])

        print(df)
        # Filter rows where the URL contains '.html'
        # df = df[df['url'].str.contains('.html')]

        # Truncate the URLs at '.html'

        df['url'] = df['url'].str.replace(':80', '', regex=False)
        df['url'] = df['url'].str.replace('http:', 'https:', regex=False)
        df['url'] = df['url'].str.split('?').str[0]
        df['url'] = df['url'].str.replace('/%7BuserImage%7D', '', regex=False)


        # #Remove the specified pattern "%20 ... 2525252F"
        # pattern_to_remove = r'%20.*2F(?=.*2F)'
        # df['url'] = df['url'].apply(lambda x: re.sub(pattern_to_remove, '', x, flags=re.DOTALL | re.IGNORECASE))

        # Remove URLs containing "news/%"
        # df = df[~df['url'].str.contains('news/%')]
        df = df[~df['url'].str.contains("/news/pro/")]
        df['url'] = df['url'].str.rstrip("/")

        # Remove duplicates
        df.drop_duplicates(subset=['url'],inplace=True)
        print(df)
        # Write the unique URLs to a CSV file
        df.to_csv(output_filename, index=False)
        # for p in paragraphs:
        #     print(p.text)
        # You can also use Scrapy's built-in selectors instead of BeautifulSoup
        # For example, to extract all text from a page:
        # text = ' '.join(response.xpath('//text()').extract())
        # print(text)

        # Here you would define your logic to process the extracted data,
        # save it to a file, store it in a database, or anything else

# Main script to start the crawler
def run_spider():
    process = CrawlerProcess()

    process.crawl(MySpider)
    process.start()  # the script will block here until the crawling is finished

# Run the spider
run_spider()
