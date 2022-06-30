import requests, os, 
from dotenv import load_dotenv

LAMBDA_ENDPOINT = os.environ.get("LAMDBA_ENDPOINT")
TRAINING_PASSWORD= os.environ.get("TRAINING_PASSWORD")

print(URL)
print("Testing training endpoint from AWS Lambda")

r = requests.post(url=LAMDBA_ENDPOINT, json={"password": TRAINING_PASSWORD})
print(r)
print(r.text)
