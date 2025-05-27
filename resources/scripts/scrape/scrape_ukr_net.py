import csv
import logging
import sys
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from scrape.news_scraper import scrape
import time
import concurrent.futures
import threading

BATCH_SIZE=2500
NUM_THREADS=8
lock = threading.Lock()

# Set up logging to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example of logging
def log_message():
    logging.info("This is a log message from my module!")

# Open a file to redirect print output
file_output = open('scrape/ukr_net.log', 'w')
# Save the current stdout to restore later
original_stdout = sys.stdout
# Redirect print statements to a file
sys.stdout = file_output

def to_absolute_url(url: str) -> str:
    return f"https://ukr.net{url}" if url.startswith('/') else url

def get_ukr_net_news(category: str) -> dict[list[tuple[str,str,str]], set[str]]:
    URL = f"https://www.ukr.net/news/{category}.html"


    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(URL, timeout=60000)
        time.sleep(1)

        last_height = page.evaluate("document.body.scrollHeight")
        while True:
            page.mouse.wheel(0, 15000)
            time.sleep(2)
            page.mouse.wheel(0, 15000)
            time.sleep(2)
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # Get the page content and parse it with BeautifulSoup
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text and href from each element with class 'im-tl_a'
        elements = soup.find_all(class_='im-tl_a')
        sources = {element.get_text().lstrip('(').rstrip(')') for element in soup.find_all(class_='im-pr_a')}
        headlines = [(to_absolute_url(element['href']), element.get_text(), category) for element in elements]

        browser.close()
    
    return {
        "headlines" : headlines,
        "sources": sources
    }
    
categories = {
    'world': 'Світ',
    'russianaggression': 'Війна',
    'politics': 'Політика', 
    'economics': 'Економіка',
    'criminal': 'Кримінал',
    'society': 'Суспільство',
    'technologies': 'Технології',
    'science': 'Наука',
    'auto': 'Авто',
    'sport': 'Спорт',
    'health': "Здоров'я",
    'show_business': 'Культура',
    'curious': 'Курйози',
    'food': 'Кулінарія',
    'sadgorod': 'Сад-город',
    # 'companies': 'Реклама'
}


def process_category(category):
    news = get_ukr_net_news(category)
    return category, news['sources'], news['headlines']

sources = set()
headlines = []

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_category, category): category for category in categories.keys()}
    
    for future in concurrent.futures.as_completed(futures):
        category = futures[future]
        try:
            category, category_sources, category_headlines = future.result()
            sources |= category_sources
            headlines.extend(category_headlines)
            logging.info(f"Category: {category}")
            logging.info(f"Total number of news in category {category}: {len(category_headlines)}")
        except Exception as exc:
            logging.error(f"Category {category} generated an exception: {exc}")

src_results = pd.DataFrame(sources, columns=['Source'])
src_results.to_csv(r'scrape/training_data/sources.csv', mode='w', index=False, encoding='utf-8', header=False, quoting=csv.QUOTE_ALL)

headlines_results = pd.DataFrame(headlines, columns=['URL', 'Title', 'Category'])
headlines_results.to_csv(r'scrape/training_dataheadlines.csv', mode='a', index=False, encoding='utf-8', header=False, quoting=csv.QUOTE_ALL)

processed_headlines = []
total_headlines = len(headlines)
    
chunk_size = len(headlines) // NUM_THREADS
chunks = [headlines[i:i + chunk_size] for i in range(0, len(headlines), chunk_size)]

def process_headline(headline, index, total):
    href = headline[0]
    title = headline[1]
    category = categories[headline[2]]
    article = scrape(href, join=True)
    if article:
        article = article["content"]
        # i = (index + 1) % chunk_size
        percentage = ((index + 1) / total) * 100
        logging.info(f"% Scraped {index+1}/{total} news ({percentage:.2f}%)")
        if len(article) > 100:
            return (href, title, category, article)
    return None    

def process_chunk(chunk, start_index, total_headlines):
    batch = []
    for i, headline in enumerate(chunk):
        result = process_headline(headline, start_index + i, total_headlines)
        if result:
            batch.append(result)
            processed_headlines.append(result)
    return batch


with open(r'scrape/training_data/news.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    # writer.writerow(['URL', 'Title', 'Category', 'Content'])    
    batch = []
    for i, headline in enumerate(headlines):
        result = process_headline(headline, i, total_headlines)
        if result:
            batch.append(result)
            processed_headlines.append(result)
        
        if len(batch) == BATCH_SIZE:
            writer.writerows(batch)
            batch = []

    # Write any remaining items in the batch
    if batch:
        writer.writerows(batch)
file.close()

# df_results = pd.DataFrame(processed_headlines, columns=['URL', 'Title', 'Category', 'Content'])
# df_results.to_csv(r'news/training_data/news.csv', mode='w', index=False, encoding='utf-8', header=False, quoting=csv.QUOTE_ALL)