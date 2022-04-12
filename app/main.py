import os, requests, sys, joblib

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
    DB_NAME = os.environ.get('MONGO_DB_NAME', None)
    CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", None)

    #connect to mongodb
    client = MongoClient(CONNECTION_STRING)
    
    collection = client[DB_NAME]['weather_data']
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))

    # need to research values to be used
    # df['pressure'] = pd.to_numeric(df['pressure'])
    # df['temperature'] = pd.to_numeric(df['temperature'])
    # df = df.sort_values('Observation Value')
    print(df)

    y = df["label"]
    X = df.drop("label", axis=1)
  
    #train with sklearn
    X_train, X_test, y_train, y_test = train_test_split(df, test_size=0.3)
    dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)
    
    cm = confusion_matrix(y_test, dtree_predictions)
    score = dtree_model.score(y_test,  d_tree.predictions()) #get model accuracy

    #create model file
    joblib.dump(dtree_model, MODEL_FILE)

    #save to mongodb
    ml_collection = client[DB_NAME]["ml"]
    with open(MODEL_FILE, "rb") as f:
        encoded = Binary(f.read())
    ml_collection.insert_one({
        "filename": MODEL_FILE, 
        "file": encoded, 
        "description": "ML Model", 
        "accuracy": score,
        "confusion_matrix": str(cm) 
        })
    
    return "Successful!"


    

