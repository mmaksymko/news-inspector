import service.fakes_service as fakes_service
import service.propaganda_service as propaganda_service
import service.clickbait_service as clickbait_service
import service.genres_service as genres_service
from newspaper import Article

def fake_news_detection(article: Article):
    return fakes_service.ml_process(article)

def database_fake_news_detection(article: Article):
    return fakes_service.db_process(article)

def propaganda_detection(article: Article):
    return propaganda_service.process(article)

def clickbait_detection(article: Article):
    return clickbait_service.process(article)

def category_classification(article: Article):
    return genres_service.process(article)

def add_fake(fake: str):
    return fakes_service.add_fake(fake)
