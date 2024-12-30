from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel

Base = declarative_base()

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: float
    Parch: int
    Fare: float
    Embarked: str


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