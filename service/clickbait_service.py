import logging
from models.base.models import models
from models.base.model_type import ModelType
from newspaper import Article

from models.const.clickbait import CLICKBAITS
from service.analytics_service import get_or_create_article
from repositories.clickbait_repository import save_clickbait

model = models[ModelType.CLICKBAIT]

def process(article: Article) -> str:
    result = model.infer(article.title or article.text)
    
    save_clickbait(get_or_create_article(article), result)
    
    logging.info(f"Clickbait result: {result}")
    return format_output(result)

def format_output(result: float) -> str:
    verdict = model.get_verdict(result)
    info = CLICKBAITS[verdict]
    probability = result if verdict else 1 - result
    return f"Вердикт: {info.name} {info.emoji}\n\nОпис: {info.desсription}\n\nЙмовірність: {probability:.2%}"

