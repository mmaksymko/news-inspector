import hashlib
from newspaper import Article as NewsArticle

from entities.analysis import Article, ArticleURL
from repositories.analytics_repository import find_by_hash, save_article, save_article_url

def compute_hash(title: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((title or '').encode('utf-8'))
    h.update((text or '').encode('utf-8'))
    return h.hexdigest()

def get_or_create_article(news_article: NewsArticle) -> Article:
    hash_value = compute_hash(news_article.title, news_article.text)
    article = find_by_hash(hash_value)
    if not article:
        article = Article(title=news_article.title, text=news_article.text, hash=hash_value)
        article = save_article(article)
        if news_article.url:
            save_article_url(ArticleURL(article_id=article.id, url=news_article.url))
    return article
