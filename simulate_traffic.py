import requests
import time

URL = "http://localhost:8000/predict"

def send_tests():
    samples = [
        "URGENT: Your account is locked! Click here.",
        "Hey, are we still meeting for coffee later?",
        "WINNER! Claim your $1000 gift card now.",
        "The report is due by Friday afternoon."
    ]
    for text in samples:
        try:
            res = requests.post(URL, json={"text": text})
            print(f"Text: {text[:30]}... -> Result: {res.json()['label']}")
        except Exception as e:
            print(f"Connection Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    send_tests()