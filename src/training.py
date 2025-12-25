import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import yaml
import os

def train_model(X_train, y_train, vectorizer):
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # --- MLFLOW ENVIRONMENT CHECK ---
    if os.environ.get('GITHUB_ACTIONS'):
        mlflow.set_tracking_uri("file:///tmp/mlruns")
    else:
        # Locally, it will use your Dockerized MLflow server
        mlflow.set_tracking_uri(config['mlflow'].get('tracking_uri', "http://localhost:5000"))
    
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run() as run:
        # Initialize and Train Model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Log Metrics (Always good for the UI)
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log Model & Vectorizer to MLflow
        mlflow.sklearn.log_model(model, config['mlflow']['model_name'])
        mlflow.sklearn.log_model(vectorizer, config['mlflow']['vectorizer_name'])
        
        # Ensure directory exists for local fallback
        os.makedirs(os.path.dirname(config['artifacts']['model_path']), exist_ok=True)
        
        # Save local model for fallback
        joblib.dump(model, config['artifacts']['model_path'])
        
        print(f"Training Complete. Run ID: {run.info.run_id}")
        return {"run_id": run.info.run_id, "model": model}