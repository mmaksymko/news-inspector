from datetime import datetime, timedelta
from repositories import admin_repository


def is_admin(handle: str) -> bool:
    return admin_repository.is_admin(handle)

def get_stats(days: int = 30):
    stats = admin_repository.get_stats(datetime.now()-timedelta(days=days))
    
    return stats
