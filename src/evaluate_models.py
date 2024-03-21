import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score # Example metric

def evaluate_models():
    # Load models
    model_rf = joblib.load('models/model_randomforest.pkl')
    model_lr = joblib.load('models/model_logreg.pkl')

    # Load evaluation data
    eval_data = pd.read_csv('data/features.csv')
    X_eval = eval_data.drop('target', axis=1)
    y_eval = eval_data['target']

    # Evaluate models
    predictions_rf = model_rf.predict(X_eval)
    predictions_lr = model_lr.predict(X_eval)
    
    accuracy_rf = accuracy_score(y_eval, predictions_rf)
    accuracy_lr = accuracy_score(y_eval, predictions_lr)

    # Save metrics
    metrics = {
        'accuracy_rf': accuracy_rf,
        'accuracy_lr': accuracy_lr
    }
    with open('reports/metrics_report.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate_models()
