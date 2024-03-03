from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.firefox.service import Service
serv = Service(executable_path='geckodriver')
opti = webdriver.FirefoxOptions()
# Set up the Selenium browser instance (e.g., with Chrome)
driver = webdriver.Firefox(service=serv, options=opti)
# Configure the Selenium WebDriver

# Start the browser and open a website
start_url = 'https://web.archive.org/web/*/https://www.cnbc.com/2024/01/04/*'  # Replace with the actual start URL
driver.get(start_url)

def scrape_current_page(driver):
    # Find all the rows on the current page
    rows = driver.find_elements(By.CSS_SELECTOR, 'tr.even')
    data = []
    for row in rows:
        # Extract the URL and the date from each row
        url = row.find_element(By.CSS_SELECTOR, 'td.url a').get_attribute('href')
        date = row.find_element(By.CSS_SELECTOR, 'td.dateFrom').text
        data.append({'url': url, 'date': date})
    return data

try:
    WebDriverWait(driver, 10).until(
        lambda d: d.find_element(By.ID, "query-summary").value_of_css_property("opacity") == "1"
    )
    # Scrape data from the start page
    data = scrape_current_page(driver)
    print(data)  # Or do something with the scraped data
    counter = 1
    # Loop to navigate to the next page and keep scraping
    while True:
        try:
            counter += 1
            print('\npage:',counter)
            # Attempt to find and click the "Next" button
            next_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-dt-idx="next"]')))
            
            if next_button.get_attribute("aria-disabled") == "true":
                print('######################')
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
            print(new_data)  # Or do something with the scraped data
            data.extend(new_data)

        except NoSuchElementException:
            # If no "Next" button is found, we've reached the last page
            print("Reached the last page.")
            break
        except TimeoutException:
            # If the page took too long to load or the "Next" button wasn't clickable
            print("Timed out waiting for page to load or 'Next' button to become clickable.")
            break

finally:
    # Close the browser window
    driver.quit()