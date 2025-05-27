import csv
import os
import sys
from bs4 import BeautifulSoup
import orjson as json
import re
from typing import Dict, List
import csv
import concurrent.futures

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'news')))
from scrape.news_scraper import scrape

URL = '<URL>'.removeprefix('https://').removeprefix('http://').removeprefix('www.')
allowed_links = rf"((?:{URL})|(?:goo\.gl))"

def extract_hrefs(messages: List[Dict[str, str]], regex = rf'^http(s)?://{allowed_links}/(?!donate)') -> List[str]:
    regex = re.compile(regex)
    hrefs = []
    for message in messages:
        text = message.get('text', '')
        for token in text:
            if isinstance(token, dict):
                if token.get('type') == 'link':
                    link = token.get('text')
                elif token.get('type') == 'text_link':
                    link = token.get('href')                    
                else:
                    continue
                # print(link)
                if regex.findall(link):
                    hrefs.append(link)
    return hrefs

def get_tags(soup: BeautifulSoup):
    tags = []
    for tag in soup.select('.c-tags__list li:not(.c-tags__title) a'):
        tags.append(tag.get_text(strip=True))
    return tags

def get_lead(soup: BeautifulSoup):
    element = soup.select_one('.s-content .o-lead')
    return element.get_text(strip=True) if element else ''

with open('tg\hro-tg.json', 'rb') as file:
    data = json.loads(file.read())

messages: List[Dict[str, str]] = data.get('messages', [])

print(len(messages))
links = extract_hrefs(messages)[::-1]
print(len(links))
links = list(dict.fromkeys(links))
print(len(links))


def process_link(link):
    article = scrape(link, [(get_lead, 'lead'),(get_tags, 'tags')])
    if article:
        return (link, article["title"], article["lead"], article["content"].replace('\n', ' '), article["tags"])
    return None

batch_size = 50
batch = []

file = URL.split('.')[-1]
with open(f'scrape/training_data/{file}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_link, link) for link in links]
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing links")):
            result = future.result()
            if result:
                batch.append(result)
                if len(batch) == batch_size:
                    writer.writerows(batch)
                    batch.clear()
        if batch:
            writer.writerows(batch)