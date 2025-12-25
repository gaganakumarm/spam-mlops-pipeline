import mlflow
from sklearn.metrics import f1_score, accuracy_score

def evaluate(cv_info, X_test_vec, y_test):
    model = cv_info['model']
    predictions = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, pos_label='spam')
    
    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    with mlflow.start_run(run_id=cv_info['run_id']):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)