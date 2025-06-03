import logging
import os
from newspaper import Article

from models.base.models import models
from models.base.model_type import ModelType
from models.const.fakes import FAKES_ML
import service.pinecone_service as ps
from service.openai_service import analyze_fake, extract_claims
from service.analytics_service import get_or_create_article
from repositories.fakes_repository import save_fake_ml, save_fake_db

MAX_CLAIMS = os.getenv("MAX_CLAIMS", 10)
VERDICT_EMOJI = {
    False: "✅",
    True: "❌"
}
model = models[ModelType.FAKE_NEWS]

def ml_process(article: Article) -> str:
    result = model.infer(article.title or article.text)
    
    save_fake_ml(get_or_create_article(article), result)
    
    logging.info(f"Fakes result: {result}")
    return format_ml_output(result)

async def db_process(article: Article) -> str:
    claims = [article.title]
    if article.text:
        claims += extract_claims(article.text, MAX_CLAIMS)

    results = await ps.find_similar_batch(claims) if len(claims) > 1 else ps.find_similar(claims[0])
    logging.info(f"DB fakes results: {results}")

    result = analyze_fake(claims, results)
    
    save_fake_db(get_or_create_article(article), result)

    logging.info(f"DB fakes result: {result}")    
    return f'{VERDICT_EMOJI[result["verdict"]]} {result["message"]}'

def add_fake(fake: str) -> None:
    ps.upsert(fake)
    
def format_ml_output(result: float) -> str:
    verdict = model.get_verdict(result)    
    result = result if verdict else 1 - result
    info = FAKES_ML[verdict]
    
    return (
        f"Вердикт: {info.name} {info.emoji}\n\n"
        f"Опис: {info.description}\n\n"
        f"Ймовірність: {result:.2%}"
    )
