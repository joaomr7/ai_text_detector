import os
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging

from src.paths import ARTIFACTS_PATH
from src.utils import save_object, evaluate_model_performance, format_model_metrics

MODEL_BEST_LIMIAR = 0.55

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(ARTIFACTS_PATH, 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data.')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1],  test_array[:, -1],
            )

            model_parameters = {
                'solver': 'lbfgs',
                'penalty': 'l2',
                'C' : 19.312640661491187,
                'class_weight': 'balanced',
                'fit_intercept': False,
                'max_iter' : 2000, 
                'random_state' :42
            }

            model = LogisticRegression(**model_parameters)

            logging.info('Initiate model training.')
            model.fit(X_train, y_train)

            logging.info('Testing model on training data...')
            train_metrics = evaluate_model_performance(model, MODEL_BEST_LIMIAR, X_train, y_train)
            logging.info('Train result: ' + format_model_metrics(*train_metrics))

            logging.info('Testing model on testing data...')
            test_metrics = evaluate_model_performance(model, MODEL_BEST_LIMIAR, X_test, y_test)
            logging.info('Test result: ' + format_model_metrics(*test_metrics))
            
            logging.info('Saving model...')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info('Model saved.')

            return format_model_metrics(*train_metrics), format_model_metrics(*test_metrics)

        except Exception as e:
            raise CustomException(e)
        
    