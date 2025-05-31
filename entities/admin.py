from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from config.sql import Base

class Admin(Base):
    __tablename__ = 'admin'

    id = Column(Integer, primary_key=True, autoincrement=True)
    handle = Column(String, unique=True, nullable=False)