import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@step(enable_cache=False)
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Started data cleaning process.")

    preprocessor = DataPreProcessor(data)

    try:
        cleaned_data = preprocessor.clean_data()

        logging.info(f"Data cleaning completed. Shape of cleaned data: {cleaned_data.shape}")
        return cleaned_data
    except Exception as e:
        logging.error(f"Data cleaning failed with error: {e}")
        raise e