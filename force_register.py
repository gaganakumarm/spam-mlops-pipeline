import mlflow
import joblib
import os

# 1. Connect to the MLflow running in Docker
mlflow.set_tracking_uri("http://localhost:5000")

model_name = "logistic_regression"
vec_name = "tfidf_vectorizer"

# 2. Create the Experiment if it's missing
if not mlflow.get_experiment_by_name("spam_classification"):
    mlflow.create_experiment("spam_classification")
mlflow.set_experiment("spam_classification")

print("🛰️ Registering local artifacts to MLflow...")

with mlflow.start_run() as run:
    # Load your locally saved pkl files
    model = joblib.load("artifacts/model/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer/vectorizer.pkl")
    
    # Log them to this new run
    mlflow.sklearn.log_model(model, model_name)
    mlflow.sklearn.log_model(vectorizer, vec_name)
    
    run_id = run.info.run_id
    
    # 3. Force them into the Registry and set to Production
    for name in [model_name, vec_name]:
        model_uri = f"runs:/{run_id}/{name}"
        mv = mlflow.register_model(model_uri, name)
        
        # Set the alias
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(name, "production", mv.version)
        print(f"{name} (Version {mv.version}) is now 'production'")

print("\nAll set! Now restart your API to pick up the new tags.")