from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options

from pickler import Pickler

class Harvester:
    url: str = "https://hypotactic.com/latin/index.html?Use_Id="

    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")
        self.my_pickler = Pickler()

    def run(self, slug: str, harvest: bool = False):
        # If we need to harvest, or no existing file is found, do a harvest
        if (harvest or not self.my_pickler.exists(slug)):
            return self.scrape(slug)
        else:
            return self.my_pickler.read(slug)

    def scrape(self, slug: str):
        print(f'Scraping {slug}.')
        driver = webdriver.Firefox(options=self.options)
        driver.get(f"{self.url}{slug}")

        # Wait for the JavaScript to populate the data
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.line"))
            )
            # We want i/v
            checkbox = driver.find_element(By.ID, "jv")
            checkbox.click()

            # Turn off macrons
            checkbox = driver.find_element(By.ID, "macrons")
            checkbox.click()

            # Now you can scrape the page as the element has appeared
            self.my_pickler.save(slug, driver.page_source)
            return driver.page_source

        finally:
            driver.quit()  # Make sure to quit the driver to free resources
