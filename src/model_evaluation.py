import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelEvalutionStrategy(ABC):
    @abstractmethod
    def evalute_model(self, model, X_test, y_test, scalar_y) -> Dict[str, float]:
        pass

class RegressionModelEvaluationStrategy(ModelEvalutionStrategy):
    def evalute_model(self, model, X_test, y_test, scalar_y):
        y_pred = model.predict(X_test)

        y_test_respahed = y_test.reshape(-1, 1)
        y_pred_respahed = y_pred.reshape(-1, 1)

        y_test_inversed = scalar_y.inverse_transform(y_test_respahed)
        y_pred_inversed = scalar_y.inverse_transform(y_pred_respahed)

        y_test_inversed = y_test_inversed.flatten()
        y_pred_inversed = y_pred_inversed.flatten()

        mse = mean_squared_error(y_test_inversed, y_pred_inversed)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inversed, y_pred_inversed)
        r2 = r2_score(y_test_inversed, y_pred_inversed)

        logging.info("Calculating evaluation metrics.")
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

class ModelEvaluator:
    def __init__(self, strategy: ModelEvalutionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvalutionStrategy):
        self._strategy = strategy

    def evaluate(self, model, X_test, y_test, scalar_y) -> Dict[str, float]:
        return self._strategy.evalute_model(model, X_test, y_test, scalar_y)