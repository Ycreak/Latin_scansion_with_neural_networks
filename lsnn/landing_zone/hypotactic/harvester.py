from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import time

class Harvester:
    def __init__(self):
        self.options = Options()
        self.options.headless = True

    def run(self, url):
        driver = webdriver.Firefox(options=self.options)
        driver.get(url)

        # Wait for the JavaScript to populate the data
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.line"))
            )
            # Disable macrons by clicking on the checkbox
            # checkbox = driver.find_element_by_id("macrons")
            checkbox = driver.find_element(By.ID, "macrons")
            checkbox.click()
            time.sleep(2)  # Wait for 5 seconds for changes to apply
            return driver.page_source

        finally:
            driver.quit()  # Make sure to quit the driver to free resources
