import requests, random, os
from dotenv import load_dotenv

load_dotenv()
HOSTNAME = os.environ.get("LAMBDA_ENDPOINT") 
TRAINING_PASSWORD = os.environ.get("TRAINING_PASSWORD")
URL= f"{HOSTNAME}"

print(URL)
print("Testing training endpoint")

r = requests.post(url=URL, json={"password": TRAINING_PASSWORD})
print(r, r.content)        