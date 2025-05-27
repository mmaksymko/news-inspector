from sqlalchemy.orm import Session

from entities.analysis import Article, FakeResult, DBFakeResult
from repositories.category_repository import get_category
from repositories.analytics_repository import create_analysis_result
from sql import with_session

FAKE_ML = 'fake_ml'
FAKE_DB = 'fake_db'

@with_session
def save_fake_ml(article: Article, score: float, session: Session=None) -> None:
    ar = create_analysis_result(article.id, FAKE_ML)
    session.add(FakeResult(analysis_id=ar.id, percentage_score=score))
    session.commit()
 
@with_session
def save_fake_db(article: Article, output: dict, session: Session=None) -> None:
    message = output["message"]
    verdict = output["verdict"]
    ar = create_analysis_result(article.id, FAKE_DB)
    session.add(DBFakeResult(analysis_id=ar.id, message=message, verdict=bool(verdict)))
