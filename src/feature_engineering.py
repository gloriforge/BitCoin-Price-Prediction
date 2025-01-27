import joblib
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler

# Abstract class for Feature Engineering strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete class for calculating SMA, EMA< RSI, and other features
class TechnicalIndicators(FeatureEngineeringStrategy):
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Calculate SMA, EMA, RSI
        df['SMA_20'] = df['CLOSE'].rolling(window=20).mean()
        df['SMA_50'] = df['CLOSE'].rolling(window=50).mean()
        df['EMA_50'] = df['CLOSE'].ewm(span=20, adjust=False).mean()

        # Price difference features
        df['OPEN_CLOSE_diff'] = df['OPEN'] - df['CLOSE']
        df['HIGH_LOW_diff'] = df['HIGH'] - df['LOW']
        df['HIGH_OPEN_diff'] = df['HIGH'] - df['OPEN']
        df['CLOSE_LOW_diff'] = df['CLOSE'] - df['LOW']

        # Lagged features
        df['OPEN_lag1'] = df['OPEN'].shift(1)
        df['CLOSE_lag1'] = df['CLOSE'].shift(1)
        df['HIGH_lag1'] = df['HIGH'].shift(1)
        df['LOW_lag1'] = df['LOW'].shift(1)

        # Rolling statistics
        df['CLOSE_roll_mean_14'] = df['CLOSE'].rolling(window=14).mean()
        df['CLOSE_roll_std_14'] = df['CLOSE'].rolling(window=14).std()

        # Drop rows with missing values (due to rolling windows, shifts)
        df.dropna(inplace=True)

        return df
    
class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, features: list, target: str):
        pass

class MinMaxScaling(ScalingStrategy):
    def scale(self, df: pd.DataFrame, features: list, target: str):
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_scaled = scaler_X.fit_transform(df[features].values)
        y_scaled = scaler_y.fit_transform(df[[target]].values)

        joblib.dump(scaler_X, 'saved_scalers/scaler_X.pkl')
        joblib.dump(scaler_y, 'saved_scalers/scaler_y.pkl')

        return X_scaled, y_scaled, scaler_y

class FeatureEngineering:
    def __init__(self, feature_strategy: FeatureEngineeringStrategy, scaling_strategy: ScalingStrategy):
        self.feature_strategy = feature_strategy
        self.scaling_strategy = scaling_strategy
    
    def process_features(self, df: pd.DataFrame, features: list, target: str):
        df_with_features = self.feature_strategy.generate_features(df)
        X_scaled, y_scaled, scaler_y = self.scaling_strategy.scale(df_with_features, features, target)
        return df_with_features, X_scaled, y_scaled, scaler_y