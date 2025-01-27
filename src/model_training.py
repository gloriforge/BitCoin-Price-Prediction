import numpy as np
import logging
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from typing import Any

class ModelBuildingStrategy:
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, fine_tuning: bool = False) -> Any:
        pass

class LSTMModelStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, fine_tuning: bool = False) -> Any:
        logging.info("Building and training LSTM model.")

        mlflow.tensorflow.autolog()

        logging.info(f"shape of X_train: {X_train.shape}")

        #LSTM model Definition
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

        model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_squence=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        mlflow.log_metric("final_loss", history.history['loss'][-1])

        logging.info("LSTM model trained and saved.")

        return model

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def train(self, X_train: np.ndarray, y_train: np.ndarray, fine_tunning: bool = False) -> Any:
        return self._strategy.build_and_train_model(X_train, y_train, fine_tunning)