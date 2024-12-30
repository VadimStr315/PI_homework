# tests/test_app.py
import pytest
import pandas as pd
from ml_model import predictor, train_titanic_model
from models import PassengerData

def test_train_titanic_model():
    model = train_titanic_model("./Data_for_Titanic/train.csv", "titanic_model.pkl")
    assert model is not None

def test_predictor():
    passenger_data = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 29,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 211.3375,
        'Embarked': 'C'
    }
    result = predictor(passenger_data)
    assert isinstance(result, str)