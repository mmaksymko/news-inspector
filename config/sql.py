from functools import wraps
from pathlib import Path
import logging
import os
import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_FOLDER = Path("data")
DB_FOLDER.mkdir(parents=True, exist_ok=True)

DB_FILENAME = DB_FOLDER / "inspector.db"
SQL_SCRIPT = Path(__file__).parent / "resources" / "sql" / "init.sql"

if not os.path.exists(DB_FILENAME):
    conn = sqlite3.connect(DB_FILENAME)
    with open(SQL_SCRIPT, 'r', encoding='utf8') as f:
        sql_script = f.read()
        conn.executescript(sql_script)
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")
else:
    logging.info("Database already exists. Skipping initialization.")

Base = declarative_base()
engine = create_engine(f'sqlite:///{DB_FILENAME}')
Session = sessionmaker(bind=engine, expire_on_commit=False)

def with_session(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session = Session()
        try:
            result = func(*args, session=session, **kwargs)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            logging.exception("Exception in %s: %s", func.__name__, e)
            raise
        finally:
            session.close()
    return wrapper
