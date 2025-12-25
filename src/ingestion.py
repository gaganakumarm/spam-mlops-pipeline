import pandas as pd
import random
import yaml
from faker import Faker

fake = Faker()

def fetch_data(n_samples=1000):
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Generating {n_samples} synthetic messages...")
    data = []
    spam_triggers = ["WINNER", "FREE CASH", "URGENT", "ACCOUNT LOCKED", "BITCOIN", "CLAIM NOW"]
    
    for _ in range(n_samples):
        if random.random() < 0.20:
            label = "spam"
            text = f"{random.choice(spam_triggers)}! {fake.sentence()} {fake.url()}"
        else:
            label = "ham"
            text = fake.sentence()
        data.append({"label": label, "text": text})
    
    df = pd.DataFrame(data)
    df.to_csv(config['data']['raw_path'], index=False)
    print(f"Data saved to {config['data']['raw_path']}")
    return df