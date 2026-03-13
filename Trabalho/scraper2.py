import time
import os
import platform
import shutil
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

def is_valid_executable(path):
    if not os.path.isfile(path):
        return False
    if platform.system() == "Windows":
        return True
    return os.access(path, os.X_OK)

def find_chrome_executable():
    system = platform.system()
    chrome_paths = []
    if system == "Windows":
        possible_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google', 'Chrome', 'Application', 'chrome.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google', 'Chrome', 'Application', 'chrome.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google', 'Chrome', 'Application', 'chrome.exe'),
        ]
        chrome_paths.extend(possible_paths)
    elif system == "Linux":
        chrome_paths.extend(['/usr/bin/google-chrome', '/usr/bin/chromium', '/snap/bin/chromium'])
    elif system == "Darwin":
        chrome_paths.append('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')

    for path in chrome_paths:
        if is_valid_executable(path):
            return path
    return shutil.which('google-chrome') or shutil.which('chromium')

class UMinhoDSpace8Scraper:
    def __init__(self, base_url, max_items=10, output_file='scraper_results.json'):
        self.base_url = base_url
        self.MAX_ITEMS = max_items
        self.output_file = output_file
        self.ANGULAR_SETTLE_TIME = 1.0
        
        chrome_options = Options()
        chrome_path = find_chrome_executable()
        if not chrome_path:
            raise FileNotFoundError("Chrome não encontrado.")
        
        chrome_options.binary_location = chrome_path
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

    def load_existing_data(self):
        """Carrega dados já guardados para evitar duplicados."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_incremental(self, paper_info):
        """Guarda o documento no ficheiro imediatamente."""
        data = self.load_existing_data()
        data.append(paper_info)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_paper_info(self, url):
        """Extrai metadados com retry básico."""
        try:
            self.driver.get(url)
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table-striped")))
            time.sleep(self.ANGULAR_SETTLE_TIME)

            targets = {
                "dc.title": "title",
                "dc.date.issued": "year",
                "dc.identifier.doi": "doi",
                "dc.contributor.author": "authors",
                "dc.description.abstract": "abstract"
            }
            data = { "title": "N/A", "year": "N/A", "doi": "N/A", "abstract": "N/A", "authors": [], "url": url }

            rows = self.driver.find_elements(By.CSS_SELECTOR, "table.table-striped tbody tr")
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 2:
                    label, value = cols[0].text.strip(), cols[1].text.strip()
                    if label in targets:
                        key = targets[label]
                        if key == "authors": data[key].append(value)
                        else: data[key] = value

            docLink = self.driver.find_elements(By.CSS_SELECTOR, "a.btn.overflow-ellipsis.mb-1")
            data["document_link"] = docLink[0].get_attribute("href") if docLink else "N/A"
            return data
        except Exception as e:
            print(f"Erro ao extrair {url}: {e}")
            return None

    def collect_all_links(self):
        """Recolhe links com suporte a paginação."""
        paper_urls = []
        self.driver.get(self.base_url)
        
        while len(paper_urls) < self.MAX_ITEMS:
            try:
                self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "ds-listable-object-component-loader")))
                time.sleep(self.ANGULAR_SETTLE_TIME)
                
                items = self.driver.find_elements(By.CSS_SELECTOR, "a.item-list-title")
                for item in items:
                    href = item.get_attribute("href").split('?')[0]
                    if href not in paper_urls:
                        paper_urls.append(href)
                        if len(paper_urls) >= self.MAX_ITEMS: break
                
                if len(paper_urls) < self.MAX_ITEMS:
                    next_btn = self.driver.find_element(By.XPATH, "//a[@aria-label='Next']")
                    self.driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(2)
            except Exception:
                break
        return paper_urls

    def scrape(self):
        existing_urls = [d['url'].replace('/full', '') for d in self.load_existing_data()]
        all_links = self.collect_all_links()
        
        results = []
        for url in all_links:
            if url in existing_urls:
                print(f"Ignora (já existe): {url}")
                continue
                
            print(f"A processar [{len(results)+len(existing_urls)+1}/{self.MAX_ITEMS}]: {url}")
            info = self.get_paper_info(url + "/full")
            if info:
                self.save_incremental(info)
                results.append(info)
                time.sleep(0.5) # Gentileza com o servidor
        
        self.driver.quit()
        return results