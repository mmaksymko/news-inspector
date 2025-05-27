import logging
from models.base.models import models
from models.base.model_type import ModelType
from newspaper import Article
from service.analytics_service import get_or_create_article

from repositories.genre_repository import save_genres

model = models[ModelType.GENRES]

def process(article: Article) -> str:
    result: dict[str, float] = model.infer(article.text or article.title)
    
    filtered_result = {item["name"] : item["probability"] for item in result if item["probability"] > model.threshold}
    save_genres(get_or_create_article(article), filtered_result)
    
    logging.info(f"Genres result: {result}")
    return model.format_output(result)