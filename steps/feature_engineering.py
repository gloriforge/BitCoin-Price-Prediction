import logging
from typing import Tuple
import pandas as pd
import numpy as np
from zenml import step
from sklearn.preprocessing import MinMaxScaler
from src.feature_engineering import FeatureEngineering, TechnicalIndicators, MinMaxScaling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame,
    features: list = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'SMA_20', 'SMA_50', 'EMA_20', 'OPEN_CLOSE_diff',
                      'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff', 'OPEN_lag1', 'CLOSE_lag1',
                      'HIGH_lag1', 'LOW_lag1', 'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'],
    target: str = 'CLOSE'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, MinMaxScaler]:
    logging.info("Started feature engineering process.")

    try:
        feature_strategy = TechnicalIndicators()
        scaling_strategy = MinMaxScaling()

        context = FeatureEngineering(feature_strategy, scaling_strategy)

        transformed_df, X_scaled, y_scaled, scaler_y = context.process_features(df, features, target)

        logging.info(f"Feature engineering completed. Data shape: {transformed_df.shape}")
        
        return transformed_df, X_scaled, y_scaled, scaler_y
    except Exception as e:
        logging.info(f"Feature engineering failed with error: {e}")
        raise e