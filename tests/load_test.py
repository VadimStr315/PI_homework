# load_test.py
from locust import HttpUser , task, between

class User(HttpUser ):
    wait_time = between(1, 5)

    @task
    def predict_survival(self):
        passenger_data = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 29,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 211.3375,
            'Embarked': 'C'
        }
        self.client.post("/predict_survival/", json=passenger_data)