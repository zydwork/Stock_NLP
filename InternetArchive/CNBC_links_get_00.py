import datetime
import json
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service

serv = Service(executable_path='geckodriver')
opti = webdriver.FirefoxOptions()
driver = webdriver.Firefox(service=serv, options=opti)

# Configure the Selenium WebDriver
start_date = datetime.date(2018, 9, 7)
end_date = datetime.date.today()

def scrape_current_page(driver):
    # Find all <tr> elements with class 'odd' or 'even'
    rows = driver.find_elements(By.CSS_SELECTOR, 'tr.odd, tr.even')
    links_data = []
    for row in rows:
        # Within each row, find all <a> elements and extract their text
        links = row.find_elements(By.TAG_NAME, 'a')
        for link in links:
            text = link.text
            if text:  # Ensure the link text is not empty
                links_data.append(text)
    return links_data

def scrape_all_pages(driver):
    data = []
    try:
        WebDriverWait(driver, 10).until(
            lambda d: d.find_element(By.ID, "query-summary").value_of_css_property("opacity") == "1"
        )
        # Scrape data from the start page
        data = scrape_current_page(driver)
        # print(data)  # Or do something with the scraped data
        counter = 1
        # Loop to navigate to the next page and keep scraping
        while True:
            try:
                counter += 1
                print('page:',counter)
                # Attempt to find and click the "Next" button
                next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-dt-idx="next"]')))
                
                if next_button.get_attribute("aria-disabled") == "true":
                    # print('######################')
                    print("Reached the last page.")
                    break
                next_button.click()
                # Wait until the next page is loaded and the "Next" button is clickable
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-dt-idx="next"]')))
                WebDriverWait(driver, 10).until(
                lambda d: d.find_element(By.ID, "query-summary").value_of_css_property("opacity") == "1"
                )
                # Scrape data from the new current page
                new_data = scrape_current_page(driver)
                # print(new_data)  # Or do something with the scraped data
                data.extend(new_data)

            except NoSuchElementException:
                # If no "Next" button is found, we've reached the last page
                print("Reached the last page.")
                break
            except TimeoutException:
                # If the page took too long to load or the "Next" button wasn't clickable
                print("Timed out waiting for page to load or 'Next' button to become clickable.")
                break

    except Exception:
        return None

    finally:  
        unique_urls_set = set()
        for url in data:
            # Split on '?' and take the first part to remove parameters.
            clean_url = url.split('?')[0] 
            # Add to a set to ensure uniqueness.
            if '.html' in clean_url:
                unique_urls_set.add(clean_url)
        return list(unique_urls_set)


def write_to_csv(date, urls):
    filename = f"cnbc_links/cnbc_{date.strftime('%Y_%m_%d')}.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['URL'])  # Header
        for url in urls:
            csvwriter.writerow([url])  # Write each URL in a separate row.


data = []

# Iterate through each date
current_date = start_date
while current_date <= end_date:
    # Generate the URL to the Wayback Machine for the current date
    url = f'https://web.archive.org/web/*/https://www.cnbc.com/{current_date.strftime("%Y/%m/%d")}*'
    print(url)
    driver.get(url)
    # Scrape links from the current page
    links = scrape_all_pages(driver)

    write_to_csv(current_date, links)




    # # Add the data to the JSON structure
    # data.append({
    #     'date': current_date.strftime("%Y-%m-%d"),
    #     'links': links
    # })
    # Increment the date by one day
    current_date += datetime.timedelta(days=1)


driver.quit()

