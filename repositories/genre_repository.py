from entities.analysis import Article, Genre, AnalysisResultGenre
from repositories.analytics_repository import create_analysis_result
from sql import with_session
from sqlalchemy.orm import Session
from repositories.category_repository import get_or_create_category

GENRES = 'genres'

@with_session
def save_genres(article: Article, results: dict[str, float], session: Session=None):
    ar = create_analysis_result(article.id, GENRES)
    for name, score in results.items():
        genre = session.query(Genre).filter_by(name=name).one()
        session.add(AnalysisResultGenre(score=score, analysis_id=ar.id, genre_id=genre.id))
