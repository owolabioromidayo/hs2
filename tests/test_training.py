import requests, random, os
from dotenv import load_dotenv

load_dotenv()

HOSTNAME = os.environ.get("ML_ENDPOINT") 
TRAINING_PASSWORD = os.environ.get("TRAINING_PASSWORD")
URL= f"{HOSTNAME}/train?password={TRAINING_PASSWORD}"

print(URL)
print("Testing training endpoint")

r = requests.post(url=URL)
print(r.text)        
