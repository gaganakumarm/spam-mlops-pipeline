import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import yaml

def train_model(X_train, y_train, vectorizer):
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run() as run:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Log Model & Vectorizer to MLflow
        mlflow.sklearn.log_model(model, config['mlflow']['model_name'])
        mlflow.sklearn.log_model(vectorizer, config['mlflow']['vectorizer_name'])
        
        # Save local model for fallback
        joblib.dump(model, config['artifacts']['model_path'])
        
        return {"run_id": run.info.run_id, "model": model}