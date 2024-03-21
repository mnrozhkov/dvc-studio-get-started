import pandas as pd
import joblib
from PIL import Image
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


    EPOCHS = 20

    with Live(dir="reports/randomforest") as live:
                
        for i in range(EPOCHS):
            live.log_metric("mae", i * random.random())
            live.log_metric("segment_A/f1", i * random.random())
            live.next_step()

            time.sleep(1)

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
        
    
    print("Training Random Forest model - COMPLETE")

if __name__ == "__main__":
    train_randomforest()
