import logging
from models.base.models import models
from models.base.model_type import ModelType
from newspaper import Article

from service.analytics_service import get_or_create_article
from repositories.clickbait_repository import save_clickbait

model = models[ModelType.CLICKBAIT]

def process(article: Article) -> str:
    result = model.infer(article.title or article.text)
    
    save_clickbait(get_or_create_article(article), result)
    
    logging.info(f"Clickbait result: {result}")
    return model.format_output(result)

