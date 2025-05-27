from datetime import datetime

from sqlalchemy import Column, Integer, Text, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, relationship

from sql import Base

class Article(Base):
    __tablename__ = 'article'
    id = Column(Integer, primary_key=True)
    title = Column('headline', Text)
    text = Column('content', Text)
    hash = Column(Text, unique=True, nullable=False)

    url = relationship('ArticleURL', back_populates='article', uselist=False)
    analysis_results = relationship('AnalysisResult', back_populates='article')

class ArticleURL(Base):
    __tablename__ = 'article_url'
    article_id = Column(Integer, ForeignKey('article.id'), primary_key=True)
    url = Column(Text, nullable=False)
    article = relationship('Article', back_populates='url')

class Category(Base):
    __tablename__ = 'category'
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)

    analysis_results = relationship('AnalysisResult', back_populates='category')

class Genre(Base):
    __tablename__ = 'genre'
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)

    analysis_results = relationship('AnalysisResultGenre', back_populates='genre')

class PropagandaTechnique(Base):
    __tablename__ = 'propaganda_technique'
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)

    analysis_results = relationship('AnalysisResultPropaganda', back_populates='technique')

class AnalysisResult(Base):
    __tablename__ = 'analysis_result'
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('category.id'))
    article_id = Column(Integer, ForeignKey('article.id'), nullable=False)
    analysed_at = Column(DateTime, default=datetime.utcnow)

    category = relationship('Category', back_populates='analysis_results')
    article = relationship('Article', back_populates='analysis_results')

    fake_result = relationship('FakeResult', back_populates='analysis', uselist=False)
    db_fake_result = relationship('DBFakeResult', back_populates='analysis', uselist=False)
    clickbait_result = relationship('ClickbaitResult', back_populates='analysis', uselist=False)
    genres = relationship('AnalysisResultGenre', back_populates='analysis')
    propaganda = relationship('AnalysisResultPropaganda', back_populates='analysis')

class FakeResult(Base):
    __tablename__ = 'fake_result'
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_result.id'))
    percentage_score = Column(Float, nullable=False)

    analysis = relationship('AnalysisResult', back_populates='fake_result')

class DBFakeResult(Base):
    __tablename__ = 'db_fake_result'
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_result.id'))
    message = Column(Text, nullable=False)
    verdict = Column(Boolean, nullable=False)

    analysis = relationship('AnalysisResult', back_populates='db_fake_result')

class ClickbaitResult(Base):
    __tablename__ = 'clickbait_result'
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analysis_result.id'))
    percentage_score = Column(Float, nullable=False)

    analysis = relationship('AnalysisResult', back_populates='clickbait_result')

class AnalysisResultGenre(Base):
    __tablename__ = 'analysis_result_genre'
    analysis_id = Column(Integer, ForeignKey('analysis_result.id'), primary_key=True)
    genre_id = Column(Integer, ForeignKey('genre.id'), primary_key=True)
    score = Column(Float, nullable=False)

    analysis = relationship('AnalysisResult', back_populates='genres')
    genre = relationship('Genre', back_populates='analysis_results')

class AnalysisResultPropaganda(Base):
    __tablename__ = 'analysis_result_propaganda'
    analysis_id = Column(Integer, ForeignKey('analysis_result.id'), primary_key=True)
    technique_id = Column(Integer, ForeignKey('propaganda_technique.id'), primary_key=True)
    score = Column(Float, nullable=False)

    analysis = relationship('AnalysisResult', back_populates='propaganda')
    technique = relationship('PropagandaTechnique', back_populates='analysis_results')
