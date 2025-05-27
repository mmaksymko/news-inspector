import logging
from models.base.models import models
from models.base.model_type import ModelType
from newspaper import Article
import service.pinecone_service as ps
from service.openai_service import analyze_fake, extract_claims
from service.analytics_service import get_or_create_article

from repositories.fakes_repository import save_fake_ml, save_fake_db

model = models[ModelType.FAKE_NEWS]

def ml_process(article: Article) -> str:
    result = model.infer(article.title or article.text)
    
    save_fake_ml(get_or_create_article(article), result)
    
    logging.info(f"Fakes result: {result}")
    return model.format_output(result)

def db_process(article: Article) -> str:
    claims = [article.title]
    if article.text:
        claims += extract_claims(article.text)

    results = [ ps.find_similar(claim) for claim in claims ]

    result = analyze_fake(claims, results)
    
    save_fake_db(get_or_create_article(article), result)

    logging.info(f"DB fakes result: {result}")    
    return result["message"]

def add_fake(fake: str) -> None:
    ps.upsert(fake)