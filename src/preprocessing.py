import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import yaml

def preprocess():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config['data']['raw_path'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save local artifact for fallback
    joblib.dump(vectorizer, config['artifacts']['vectorizer_path'])
    
    return X_train_vec, X_test_vec, y_train, y_test