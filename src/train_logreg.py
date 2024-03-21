import pandas as pd
import joblib
import random
from sklearn.linear_model import LogisticRegression 
import time

from pathlib import Path
from dvclive import Live


def train_logreg():

    print("Training Logistic Regression model - START")

    # Load the prepared data
    data = pd.read_csv('data/features.csv')

    # Extract features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Train the model
    model = LogisticRegression()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, 'models/model_logreg.pkl')

    with Live() as live:
        live.log_artifact(
            path='models/model_logreg.pkl',
            type="model",
            name="logreg",
            labels=["lr"],
        )

        live.log_metric("f1", random.random(), plot=False)
        live.log_metric("mae", random.random(), plot=False)

    print("Training Logistic Regression model - COMPLETE")

if __name__ == "__main__":
    train_logreg()
