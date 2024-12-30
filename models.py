from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PassengerModel(Base):
    __tablename__ = "passengers"

    id = Column(Integer, primary_key=True, index=True)
    Pclass = Column(Integer)
    Sex = Column(String)
    Age = Column(Float)
    SibSp = Column(Float)
    Parch = Column(Integer)
    Fare = Column(Float)
    Embarked = Column(String)