import requests
import pandas as pd
import os

def fetch_crypto_data(api_url, api_key):
    response = requests.get(
        api_url,
        params={
            "market": "cadli",
            "instrument": "BTC-USD",
            "limit": 5000,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "api_key": api_key
        },
        headers={
            "Content-type": "application/json; charset=UTF-8"
        }
    )

    if response.status_code == 200:
        print('API Connection Successful! \nFetching the data...')

        data = response.json()
        data_list = data.get("Data")

        df = pd.DataFrame(data_list)

        df['DATE'] = pd.to_datetime(df['TIMESTAMP'], unit='s')

        return df
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")