import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def train_titanic_model(train_data_path, model_path):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        train_df = pd.read_csv(train_data_path)

        train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
        train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
        train_df['Embarked'].fillna('S', inplace=True)
        train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = train_df[features]
        y = train_df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        joblib.dump(model, model_path)
        print("Модель сохранена.")

    return model

def predict_survival_pass(model, passenger_data):
    passenger_df = pd.DataFrame([passenger_data])
    
    passenger_df['Sex'] = passenger_df['Sex'].map({'male': 0, 'female': 1})
    passenger_df['Embarked'] = passenger_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    passenger_df['Age'].fillna(passenger_df['Age'].mean(), inplace=True)
    
    probability = model.predict_proba(passenger_df)[:, 1]
    return probability[0]

def predictor(passenger_data):
    model_path = "titanic_model.pkl"
    model = train_titanic_model("./Data_for_Titanic/train.csv", model_path)
    survival_probability = predict_survival_pass(model, passenger_data)
    print(f"Вероятность выживания пассажира: {survival_probability:.2f}")
    return f"{survival_probability:.2f}"

if __name__ == "__main__":
    passenger_data = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 29,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 211.3375,
        'Embarked': 'C'
    }
    predictor(passenger_data)