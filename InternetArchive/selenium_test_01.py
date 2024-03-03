from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
import time

serv = Service(executable_path='geckodriver')
opti = webdriver.FirefoxOptions()
opti.set_preference("permissions.default.image", 2)
opti.set_preference("javascript.enabled", True)
opti.set_preference("dom.ipc.plugins.enabled.libflashplayer.so", False)

# Set up the Selenium browser instance (e.g., with Chrome)
driver = webdriver.Firefox(service=serv, options=opti)

# # Navigate to the login page
# driver.get('https://www.example.com/login')

# # Input the username and password and submit the form
# username = driver.find_element(By.ID, 'username')
# password = driver.find_element(By.ID, 'password')

# username.send_keys('your_username')
# password.send_keys('your_password')

# # You might need to adjust the selectors based on the form you're automating
# login_button = driver.find_element(By.ID, 'login_button')
# login_button.click()

# # Wait for the login to complete (you may need to increase the sleep time)
# time.sleep(5)

# Now you can navigate to the page you want to scrape
driver.get('https://www.msn.com/en-us/money/insurance/california-insurance-crisis-deepens-as-providers-pull-out-of-state/ar-BB1hfdEr')
WebDriverWait(driver, 3).until(
lambda d: d.execute_script('return document.readyState') == 'complete')
# Scrape your data
page_source = driver.page_source

#print(page_source)

# Initialize BeautifulSoup and parse the page source with the desired parser, e.g., 'html.parser' or 'lxml'
soup = BeautifulSoup(page_source, 'html.parser')



# Now you can use BeautifulSoup to find elements on the page
# Example: Find the first h1 tag
# first_h1 = soup.find('h1')

# Example: Find all paragraph tags
paragraphs = soup.find_all('p')
print(soup)
text = soup.get_text(separator='\n', strip=True)
print(text)

# Print the text of the first h1 tag
# print(first_h1.text if first_h1 else 'No h1 tag found')

# Print the text of each paragraph
for p in paragraphs:
    print(p.text)

driver.quit()