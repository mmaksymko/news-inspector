import logging
import requests
from bs4 import BeautifulSoup, Comment
from htmldate import find_date
from playwright.sync_api import sync_playwright
from scrape.bypass_cloudflare import load_and_bypass_cloudflare
from utils.string_utils import join_paragraphs, normalize_quotes, process_escape_sequences, remove_emojis
import scrape.scraper_utils as su
import cloudscraper

SCRAPER = cloudscraper.create_scraper(delay=3, browser={'custom': 'ScraperBot/1.0'})
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_article(page, deep_clean=True):
    def get_title(page, article_text):
        h1 = page.find('h1')    
        if h1 and h1.get_text():
            title = h1.get_text()
        else:
            split = article_text.split('.')
            if len(split[0]) >= 10:
                title = split[0]
            else:
                title = split[1]
        title = process_escape_sequences(title)
        if split:= title.split('\n')[0]:
            title = split
        return title

    h1s = page.find_all('h1')
    title = None
    if len(h1s) == 1:
        title = h1s[0].get_text(strip=True)

    su.clean_up(page, deep_clean)

    paragraphs_src = su.find_paragraphs(page)
    paragraphs = su.get_content(paragraphs_src)
    
    if not paragraphs:
        if deep_clean:
            return scrape_article(page, deep_clean=False)
        return None
    
    while paragraphs[-1].endswith(':'):
        paragraphs.pop()
    
    article_text = "\n".join(paragraphs)
    article_text = process_escape_sequences(article_text) 
        
    if not article_text or len(article_text.split(' ')) < 15:
        if deep_clean:
            return scrape_article(page, deep_clean=False)
        return None
    
    if not title:
        title = get_title(page, article_text)
    
    article_text = normalize_quotes(article_text)
    return {
        "title": remove_emojis(title.strip()),
        "content": remove_emojis(article_text.strip())
    }
    
def load_dynamically(url:str, timeout: int = 10000) -> str:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        
        page = browser.new_page()

        page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        page.wait_for_selector('body', timeout=timeout, state='attached')
        
        content = page.content()

        browser.close()

        return content
    return content

def scrape(url:str, *functions: list[tuple[callable, str]], join: bool = False) -> dict[str, str]:
    def scrape_cloudflare(url: str, timeout: int = 5000) -> str:
        response = load_and_bypass_cloudflare(url)
        response = response
        page = BeautifulSoup(response, 'html.parser')
        return scrape_article(page)
    
    try:
        response = SCRAPER.get(url, headers=HEADERS, timeout=5)
        
        if response.encoding == 'ISO-8859-1':
            response.encoding = 'utf-8'

        status_code = response.status_code
    
        if status_code == 403:
            result = scrape_cloudflare(url)
        if status_code != 200:
            logging.error(f"Failed to retrieve the article {status_code} - {url}")
            return None
        
        page = BeautifulSoup(response.text, 'html.parser')
        
        result = scrape_article(page)

        if not result:
            content = load_dynamically(url)
            page = BeautifulSoup(content, 'html.parser')
            result = scrape_article(page)
        if result and result['content'] == 'Verifying you are human. This may take a few seconds.':
            result = scrape_cloudflare(url)

        if join and result:
            result['content'] = join_paragraphs(result['content'])
            
        for func in functions:
            for function, name in func:
                result[name] = function(page)
          
        return result

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error at {url}: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed at {url}: {e.__class__.__name__} - {e}")
    except Exception as e:
        logging.error(f"Unexpected error at {url}: {e.__class__.__name__} - {e}")
    
    return None