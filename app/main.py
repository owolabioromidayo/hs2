import os, requests, sys, joblib, datetime

from flask import Flask, render_template, request
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from bson.binary import Binary

import pandas as pd

load_dotenv()

app = Flask(__name__)


@app.route("/train", methods=["POST"])
def train_model():
    MODEL_FILE = "model.pkl"
    MODE = os.environ.get("MODE", "prod")
    DB_NAME = os.environ.get('MONGO_DB_NAME', None)
    CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", None)

    #connect to mongodb
    client = MongoClient(CONNECTION_STRING)

    collection = None
    if MODE != "test":
        collection = client[DB_NAME]['weather_data']
    else:
        collection = client[DB_NAME]['weather_data_test']

    cursor = collection.find()
    df = pd.DataFrame(list(cursor))

    y = df["label"]
    X = df[['baro_pressure', 'ext_temp', 'humidity', 'wind_speed', 'wind_direction']]
  
    #train with sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
    
    dtree_predictions = dtree_model.predict(X_test)
    cm = confusion_matrix(y_test, dtree_predictions)
    score = dtree_model.score(X_test, y_test) #get model accuracy
    
    joblib.dump(dtree_model, MODEL_FILE) #create model file

    #save to mongodb
    ml_collection = None
    if MODE != "test":
        ml_collection = client[DB_NAME]["ml"]
    else:
        ml_collection = client[DB_NAME]["ml_test"]

    with open(MODEL_FILE, "rb") as f:
        model_bin = Binary(f.read())

    dt = datetime.datetime.now()
    ml_collection.insert_one({
        "file": model_bin, 
        "description": f"Dtree Model Snapshot at {dt}", 
        "accuracy": score,
        "confusion_matrix": str(cm),
        "datetime": dt
        })
    
    return "Successful!"


    
