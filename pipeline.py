import os
import yaml
import joblib
from src.ingestion import fetch_data
from src.validation import validate_data
from src.preprocessing import preprocess
from src.training import train_model
from src.evaluation import evaluate
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

def run_pipeline():
    for path in ["data/raw", "data/processed", "artifacts/model", "artifacts/vectorizer"]:
        os.makedirs(path, exist_ok=True)

    print("\nStarting 1000-Sample Pipeline...")
    
    # 1. Ingestion
    raw_df = fetch_data(n_samples=1000)
    
    # 2. Validation
    valid_df = validate_data(raw_df)
    
    # 3. Preprocessing
    X_train_vec, X_test_vec, y_train, y_test = preprocess()
    
    # 4. Training (Loading vectorizer to log it)
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    vectorizer = joblib.load(config['artifacts']['vectorizer_path'])
    
    cv_info = train_model(X_train_vec, y_train, vectorizer)
    
    # 5. Evaluation
    evaluate(cv_info=cv_info, X_test_vec=X_test_vec, y_test=y_test)
    print("\nPipeline Finished Successfully!")

if __name__ == "__main__":
    run_pipeline()