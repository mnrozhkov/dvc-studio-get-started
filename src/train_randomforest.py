import pandas as pd
import joblib
import random
from sklearn.ensemble import RandomForestClassifier
import time

from pathlib import Path
from dvclive import Live

def train_randomforest():

    print("Training Random Forest model - START")
    

    # Load the prepared data
    data = pd.read_csv('data/features.csv')

    # Extract features and target variable
    X = data.drop('target', axis=1)
    y = data['target']

    # Train the model
    model = RandomForestClassifier() # Example model
    model.fit(X, y)

    # Simulate GridSearch hyperparameter tuning
    errors = []
    max_leaf_nodes = [5, 50, 500, 5000]
    for nodes in max_leaf_nodes:
        error = random.random()
        errors.append(error)
        print(f"Max leaf nodes: {nodes}  \t ➡ Mean Absolute Error:  {error}")
    # Create a DataFrame from the lists
    datapoints = pd.DataFrame({
        'Max Leaf Nodes': max_leaf_nodes,
        'Error': errors
    })

    # Save the trained model
    MODEL_PATH = "models/model_randomforest.pkl"
    joblib.dump(model, MODEL_PATH)



    with Live(
        dir="reports/randomforest",
        # save_dvc_exp=True,
    ) as live:
        
        live.log_artifact(
            path=MODEL_PATH,
            type="model",
            name="randomforest",
            labels=["rf"],
        )

        live.log_plot(
            "errors_vs_leafs",
            datapoints,
            x="Max Leaf Nodes",
            y="Error",
            template="simple",
            title="Errors vs Max Leaf Nodes")

    # time.sleep(2)
    print("Training Random Forest model - COMPLETE")

if __name__ == "__main__":
    train_randomforest()
