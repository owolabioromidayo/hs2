import os, requests, sys, joblib, datetime, threading

from flask import Flask, render_template, request
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from bson.binary import Binary
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()
app = Flask(__name__)

def train():
    MODEL_FILE = "model.pkl"
    IMAGE_FILE="dtree.png"
    DB_NAME = os.environ.get('MONGO_DB_NAME', None)
    CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", None)

    #connect to mongodb
    client = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME]['weather_data']

    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    df.dropna()

    y = df["label"]
    X = df[['baro_pressure', 'ext_temp', 'humidity', 'wind_speed', 'uv']]

  
    #train with sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
    
    dtree_predictions = dtree_model.predict(X_test)
    cm = confusion_matrix(y_test, dtree_predictions)
    score = dtree_model.score(X_test, y_test) #get model accuracy
    
    joblib.dump(dtree_model, MODEL_FILE) #create model file

    #create image
    class_names = [
        "Thunderstorm with light rain",
        "Thunderstorm with rain",
        "Thunderstorm with heavy rain",
        "Thunderstorm with light drizzle",
        "Thunderstorm with drizzle",
        "Thunderstorm with heavy drizzle",
        "Thunderstorm with Hail",
        "Light Drizzle",
        "Drizzle",
        "Heavy Drizzle",
        "Light Rain",
        "Moderate Rain",
        "Heavy Rain",
        "Freezing rain",
        "Light shower rain",
        "Shower rain",
        "Heavy shower rain",
        "Light snow",
        "Snow",
        "Heavy Snow",
        "Mix snow/rain",
        "Sleet",
        "Heavy sleet",
        "Snow shower",
        "Heavy snow shower",
        "Flurries",
        "Mist",
        "Smoke",
        "Haze",
        "Sand/dust",
        "Fog",
        "Freezing Fog",
        "Clear sky",
        "Few clouds",
        "Scattered clouds",
        "Broken clouds",
        "Overcast clouds",
        "Unknown Precipitation"
    ]

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(dtree_model, 
                    feature_names=['baro_pressure', 'ext_temp', 'humidity', 'wind_speed', 'uv'],  
                    class_names=class_names,
                    filled=True)
                
    fig.savefig("dtree.png")

    #save to mongodb
    ml_collection = client[DB_NAME]["ml"]

    with open(MODEL_FILE, "rb") as f:
        model_bin = Binary(f.read())

    with open(IMAGE_FILE, "rb") as f:
        image_bin = Binary(f.read())

    dt = datetime.datetime.now()
    ml_collection.insert_one({
        "file": model_bin, 
        "image-png": image_bin,
        "description": f"Dtree Model Snapshot at {dt}", 
        "accuracy": score,
        "confusion_matrix": str(cm),
        "datetime": dt,
        "n_samples_used" : len(df)
        })


@app.route("/train", methods=["POST"])
def train_model():
    if request.args.get("password") != os.environ.get("TRAINING_PASSWORD"):
        return "Unauthorized", 401
    
    thread = threading.Thread(target=train)
    thread.start()

    return "Training Model!", 200


    

