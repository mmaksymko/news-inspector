from sqlalchemy.orm import Session

from entities.analysis import Article, AnalysisResult, ClickbaitResult
from repositories.analytics_repository import create_analysis_result
from config.sql import with_session
from repositories.category_repository import get_or_create_category

CLICKBAIT = 'clickbait'

@with_session
def save_clickbait(article: Article, score: float, session: Session=None):
    ar = create_analysis_result(article.id, CLICKBAIT)
    session.add(ClickbaitResult(analysis_id=ar.id, percentage_score=score))
