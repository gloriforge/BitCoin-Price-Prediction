import os
from pymongo import MongoClient
from fetch_data import fetch_crypto_data
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

mongo_url = os.getenv('MONGO_URL')
api_url = os.getenv('API_URL')
api_key = os.getenv('API_KEY')

client = MongoClient(mongo_url)
db = client['crypto_data']
collection = db['historical_data']

try:
    latest_entry = collection.find_one(sort=[("DATE", -1)])
    if latest_entry:
        last_date = pd.to_datetime(latest_entry["DATE"]).strftime('%Y-%m-%d')
    else:
        last_date = '2011-03-27'
    
    print(f"Fetching data starting from {last_date}...")
    new_data_df = fetch_crypto_data(api_url, api_key)

    if latest_entry:
        new_data_df = new_data_df[new_data_df['DATE'] > last_date]
    
    if not new_data_df.empty:
        data_to_insert = new_data_df.to_dict(orient='records')
        result = collection.insert_many(data_to_insert)
        print(f"Inserted {len(result.inserted_ids)} new records into MongoDB.")
    else:
        print("No new data to insert.")

except Exception as e:
    print(f"An error occured: {e}")