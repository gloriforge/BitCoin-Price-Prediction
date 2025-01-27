import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
from zenml import step
import pandas as pd

load_dotenv()

mongo_url = os.getenv('MONGO_URL')

def fetch_data_from_mongodb(collection_name:str, database_name:str):
    client = MongoClient(mongo_url)
    db = client[database_name]
    collection = db[collection_name]

    try:
        logging.info(f"Fetching data from MongoDB collection. {collection_name}...")
        data = list(collection.find())

        if not data:
            logging.info("No data found in the MongoDB collection.")
        
        df = pd.DataFrame(data)

        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        
        logging.info("Data successfully fetched and converted to a DataFrame!")
        return df
    
    except Exception as e:
        logging.error(f"An error occured with fetching data from database: {e}")
        raise e
