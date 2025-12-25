import pandas as pd

def validate_data(df):
    print("Validating data...")
    # Check for missing values
    if df.isnull().values.any():
        df = df.dropna()
    
    # Ensure labels are correct
    valid_labels = ['ham', 'spam']
    df = df[df['label'].isin(valid_labels)]
    # Remove empty texts
    df = df[df['text'].str.strip().astype(bool)]
    
    print(f"Validation complete. {len(df)} rows remain.")
    return df