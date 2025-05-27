from functools import lru_cache
from sqlalchemy.orm import Session
from entities.analysis import Category

def get_or_create_category(session: Session, name: str) -> Category:
    category = get_category(session, name)
    if not category:
        category = Category(name=name)
        session.add(category)
        session.flush()
    return category

@lru_cache(maxsize=128)
def get_category(session: Session, name: str) -> Category | None:
    return session.query(Category).filter_by(name=name).one_or_none()