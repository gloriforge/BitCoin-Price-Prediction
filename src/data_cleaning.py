import logging
import pandas as pd

class DataPreProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        logging.info("DataPreprocessor intialized with data of shape: %s", data.shape)
    
    def clean_data(self) -> pd.DataFrame:
        '''
            Data cleaning by removing unnecessary columns, dropping columns with invalid or missing values and returning the cleaned DataFrame.
        '''

        logging.info("Starting data cleaning process.")

        columns_to_drop = [
            'UNIT', 'TYPE', 'MARKET', 'INSTRUMENT', 
            'FIRST_MESSAGE_TIMESTAMP', 'LAST_MESSAGE_TIMESTAMP', 
            'FIRST_MESSAGE_VALUE', 'HIGH_MESSAGE_VALUE', 'HIGH_MESSAGE_TIMESTAMP', 
            'LOW_MESSAGE_VALUE', 'LOW_MESSAGE_TIMESTAMP', 'LAST_MESSAGE_VALUE', 
            'TOTAL_INDEX_UPDATES', 'VOLUME_TOP_TIER', 'QUOTE_VOLUME_TOP_TIER', 
            'VOLUME_DIRECT', 'QUOTE_VOLUME_DIRECT', 'VOLUME_TOP_TIER_DIRECT', 
            'QUOTE_VOLUME_TOP_TIER_DIRECT', '_id' 
        ]
        logging.info("Dropping unnecessary columns.")
        self.data = self.drop_columns(self.data, columns_to_drop)

        logging.info("Dropping columns with missing values.")
        self.data = self.drop_columns_with_missing_values(self.data)

        logging.info("Data cleaning completed. Data shape after cleaning: %s", self.data.shape)
        return self.data

    def drop_columns(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:        
        logging.info("Dropping columns: %s", columns)
        return data.drop(columns=columns, errors='ignore')
    
    def drop_columns_with_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        missing_columns = data.columns[data.isnull().sum() > 0]
        if not missing_columns.empty:
            logging.info("Columns with missing values: %s", missing_columns.tolist())
        else:
            logging.info("No columns with missing values found.")
        return data.loc[:, data.isnull().sum() == 0]