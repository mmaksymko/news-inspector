import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from resources.scripts.scrape.news_scraper import scrape

df = pd.read_csv('news\companies.csv', header=None, names=['Title', 'Category'])
titles_in_df = set(df['Title'])
print("Number of titles in the dataframe:", len(titles_in_df))


STEP = 44
value = 0
results = []

while True:
    url = f'https://zaxid.net/partnerski_materiali_tag56136/newsfrom{value}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    news_titles = soup.find_all(class_='news-title')
    titles = soup.find_all(class_='title')
    
    print(f"{value} Number of news titles:", len(news_titles) + len(titles))
    if len(news_titles) + len(titles) == 6:
        break
    
    for news_title in news_titles:
        parent = news_title.find_parent('a')
        if parent and parent.has_attr('href'):
            href = parent['href']
            text = news_title.get_text(strip=True)
            results.append((text, href))
    
    for title in titles:
        parent = title.find_parent('a')
        if parent and parent.has_attr('href'):
            href = parent['href']
            text = title.get_text(strip=True)
            results.append((text, href))
    
    value += STEP

print("Number of results:", len(results))
results = [(result[1], result[0], 'Реклама', scrape(result[1])["content"].replace('\n', ' ')) for result in results if result[0] in titles_in_df]

df_results = pd.DataFrame(results, columns=['URL', 'Title', 'Category', 'Content'])

df_results.to_csv(r'news/news.csv', index=False, encoding='utf-8', header=False, quoting=csv.QUOTE_ALL)