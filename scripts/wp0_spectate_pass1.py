"""WP0 spectate smoke, pass 1: load the freeciv-web client headlessly, capture
the index and dump the visible structure to locate the running game / observe path."""
import os
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

OUT = '/home/pav/code/lechonkgameharness/artifacts/runs/wp0-spectate'
os.makedirs(OUT, exist_ok=True)

opts = Options()
opts.add_argument('-headless')
profile = '/home/pav/.firefox-selenium-profile'
os.makedirs(profile, exist_ok=True)
opts.add_argument('-profile')
opts.add_argument(profile)
os.environ['MOZ_DISABLE_CONTENT_SANDBOX'] = '1'
os.environ['MOZ_DISABLE_GFX_SANDBOX'] = '1'
driver = webdriver.Firefox(options=opts)
try:
    driver.get('http://localhost:8080')
    time.sleep(5)
    print('[title]', driver.title)
    driver.save_screenshot(os.path.join(OUT, 'index.png'))
    print('[url]', driver.current_url)
    text = driver.find_element('tag name', 'body').text
    print('[body]')
    print(text[:2000])
    links = [a.get_attribute('href') for a in driver.find_elements('tag name', 'a')]
    print('[links]', [l for l in links if l][:20])
finally:
    driver.quit()
print('[done]')
