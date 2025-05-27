import logging
from newspaper import Article

from models.base.models import models
from models.base.model_type import ModelType
from models.const.propaganda import PropagandaTechniqueInfo
from service.analytics_service import get_or_create_article
from repositories.propaganda_repository import save_propaganda

model = models[ModelType.PROPAGANDA]

def process(article: Article) -> str:
    result: list[tuple[PropagandaTechniqueInfo, float]] = model.infer(article.title or article.text)
    
    filtered_result = {info.name :score for info, score in result if score >= model.threshold}

    save_propaganda(get_or_create_article(article), filtered_result)
    
    logging.info(f"Propaganda result: {result}")
    return model.format_output(result)
