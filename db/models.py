from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///bot.db')

Base = declarative_base()


class Game(Base):
    __tablename__ = 'Game'

    id = Column(Integer, primary_key=True)
    date = Column(String)
    board = Column(String)
    winner = Column(String)


class Dataset(Base):
    __tablename__ = 'Dataset'

    id = Column(Integer, primary_key=True)
    field = Column(String)
    player = Column(String)
    move = Column(String)
    winner = Column(String)


class X(Base):
    __tablename__ = 'X'

    id = Column(Integer, primary_key=True)
    field = Column(String)
    move = Column(String)


class O(Base):
    __tablename__ = 'O'

    id = Column(Integer, primary_key=True)
    field = Column(String)
    move = Column(String)


start_db = Base.metadata.create_all(engine)
