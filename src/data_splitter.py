import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, X: np.ndarray, y: np.ndarray):
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, X: np.ndarray, y: np.ndarray):
        logging.info("Performing simple train-test split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test

class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        logging.info("Swtiching data splitting strategy.")
        self.strategy = strategy
    
    def split(self, X: np.ndarray, y: np.ndarray):
        logging.info("Splitting data using the selected strategy.")
        return self.strategy.split_data(X, y)