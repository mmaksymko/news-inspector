from utils.log_utils import log_io
from sql import with_session
from sqlalchemy.orm import Session
from entities.analysis import Article, ArticleURL, AnalysisResult
from repositories.category_repository import get_or_create_category

@log_io()
@with_session
def find_by_hash(hash: str, session = None) -> Article | None:
    return session.query(Article).filter_by(hash=hash).one_or_none()

@with_session
def save_article(article: Article, session = None):
    session.add(article)
    return article
    
@with_session
def save_article_url(url: ArticleURL, session = None):
    session.add(url)
    return url

@with_session
def create_analysis_result(article_id: int, category: str, session: Session=None):
    cat = get_or_create_category(session, category)
    ar = AnalysisResult(article_id=article_id, category_id=cat.id)
    session.add(ar)
    return ar

