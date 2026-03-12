from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

url = "https://warhammer40k.fandom.com/ru/wiki/Око_Ужаса"

options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)

driver.get(url)

time.sleep(10)

body = driver.find_element(By.TAG_NAME, "body")
text = body.text

with open("page.txt", "w", encoding="utf-8") as file:
    file.write(text)

driver.quit()
