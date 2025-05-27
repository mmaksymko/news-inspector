from datetime import datetime, timedelta
from sqlalchemy import func, select
from entities.admin import Admin
from entities.analysis import AnalysisResult, Category
from sql import with_session
from sqlalchemy.orm import Session

@with_session
def is_admin(handle: str, session: Session = None) -> bool:
    return session.query(Admin).filter_by(handle=handle).scalar() is not None

@with_session
def add_admin(handle: str, session: Session = None) -> Admin:
    admin = Admin(handle=handle)
    session.add(admin)
    return admin

@with_session
def get_stats(since: datetime = datetime.now()-timedelta(days=30), session: Session = None):
    count_alias = func.count(AnalysisResult.id).label("result_count")

    stmt = (
        select(Category.name, count_alias)
        .join(AnalysisResult, AnalysisResult.category_id == Category.id)
        .where(AnalysisResult.analysed_at > since)
          .group_by(Category.name)
        .order_by(count_alias.desc())
    )
    
    return session.execute(stmt).all()