from datetime import datetime
from sqlalchemy.orm import Session

from entities.analysis import Article, PropagandaTechnique, AnalysisResultPropaganda
from repositories.analytics_repository import create_analysis_result
from sql import with_session

PROPAGANDA = 'propaganda'

@with_session
def save_propaganda(article: Article, result: dict[str, float], session: Session=None):
    ar = create_analysis_result(article.id, PROPAGANDA)
    for name, score in result.items():
        tech = session.query(PropagandaTechnique).filter_by(name=name).one()
        session.add(AnalysisResultPropaganda(analysis_id=ar.id, technique_id=tech.id, score=float(score)))
