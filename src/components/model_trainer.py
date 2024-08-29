import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
from dataclasses import dataclass
import os
import sys
from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

@dataclass
class ModelTrainerConfig:
    #trained_model_file_path = os.path.join("artifacts", "model.pkl")
    trained_model_file_path: str = os.path.join('../../', 'artifacts', 'model.pkl' )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            model_report: Dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model name
            best_model_name = max(model_report, key=lambda k: model_report[k]['Accuracy'])
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}')
            print(f'Accuracy Score: {model_report[best_model_name]["Accuracy"]}')
            print('\n====================================================================================\n')
            logging.info(
                f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {model_report[best_model_name]["Accuracy"]}')
            logging.info('Best found model on both training and testing dataset')

            # Define the search space for Hyperopt
            space = {
                'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                'max_depth': hp.choice('max_depth', [3, 5, 10, None]),
                'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])
            }

            def objective(params):
                with mlflow.start_run(nested=True):
                    # Log the parameters
                    mlflow.log_params(params)
                    best_model.set_params(**params)

                    # Fit the model
                    best_model.fit(X_train, y_train)

                    # Predict and calculate accuracy
                    y_pred = best_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    mlflow.log_metric("accuracy", accuracy)

                    return {'loss': -accuracy, 'status': STATUS_OK}

            # Set the tracking URI and experiment name
            mlflow.set_tracking_uri("http://127.0.0.1:8080")
            mlflow.set_experiment("HP Tuning Hyperopt")

            with mlflow.start_run(run_name='Main Run'):
                # Optimize the hyperparameters using Hyperopt
                trials = Trials()
                best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

            # Log the best parameters and accuracy
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_params(best_params)
            mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)
