import csv, os
from pymongo import MongoClient
from dotenv import load_dotenv


def ingest_from_csv(filename, collection):
    try:
            with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                entry = {}
                labels = []
                for idx, row in enumerate(reader):
                    if idx == 0:
                        labels = row
                        print(labels)
                    else:
                        entry = dict()
                        for i, label in enumerate(labels):
                            if label != "label":
                                entry[label] = float(row[i])
                            else:
                                entry[label] = row[i]

                        collection.insert_one(entry)

    except FileNotFoundError:
            print('File does not exist')


def get_collection():
    CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING", None)
    client = MongoClient(CONNECTION_STRING)
    client['dataStore']['weather_data_test'].drop()
    return client['dataStore']['weather_data_test']


def print_collection(collection_name):   
    for item in collection_name.find():
        print(item)


if __name__ == "__main__":    
    load_dotenv()
    collection = get_collection()
    ingest_from_csv('fake_data.csv', collection)
    print_collection(collection)