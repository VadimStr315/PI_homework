# tests/test_regression.py
import pytest
from ml_model import predictor

@pytest.mark.parametrize("passenger_data, expected_probability", [
    ({"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 211.3375, "Embarked": "C"}, "0.97"),
    ({"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}, "0.13"),
])
def test_regression(passenger_data, expected_probability):
    result = predictor(passenger_data)
    assert result == expected_probability