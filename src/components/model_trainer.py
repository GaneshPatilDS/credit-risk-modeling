import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
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
    trained_model_file_path: str = os.path.join('../../', 'artifacts', 'model.pkl')


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

            logging.info(f'Unique values in y_train: {np.unique(y_train)}')
            logging.info(f'Unique values in y_test: {np.unique(y_test)}')

            # Step 1: Evaluate models with default parameters
            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            model_report: Dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info(f'Model Report: {model_report}')

            # Get the best model based on accuracy
            best_model_name = max(model_report, key=lambda k: model_report[k]['F1'])
            logging.info(f'Best Model Found: {best_model_name}, F1: {model_report[best_model_name]['F1']}')

            # Step 2: Perform hyperparameter tuning for the best model
            logging.info(f'Performing hyperparameter tuning for {best_model_name}')

            space = get_hyperparameter_space(best_model_name)

            def objective(params):
                # Create a new instance of the model with the given hyperparameters
                if best_model_name == "Random Forest":
                    model = RandomForestClassifier(**params)
                elif best_model_name == "XGBClassifier":
                    model = XGBClassifier(**params)
                elif best_model_name == "CatBoost Classifier":
                    model = CatBoostClassifier(**params, verbose=False)
                elif best_model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier(**params)
                elif best_model_name == "AdaBoost Classifier":
                    model = AdaBoostClassifier(**params)
                else:
                    raise ValueError(f"Model {best_model_name} not supported for hyperparameter tuning.")

                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)   # Log hyperparameters
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)  # For log loss or probabilistic evaluation

                    # Calculate accuracy and log loss
                    f1 = f1_score(y_test, y_pred)
                    loss_value = log_loss(y_test, y_pred_proba)

                    logging.info(f'f1_score': {f1}, Loss: {loss_value}')
                    mlflow.log_metrics({'f1_score': f1, 'loss': loss_value})  # Log metrics

                    return {'loss': loss_value, 'f1_score': f1, 'status': STATUS_OK}

            mlflow.set_tracking_uri("http://127.0.0.1:5003")
            mlflow.set_experiment("HP_Tuning_Hyperopt")

            with mlflow.start_run(run_name='Main Run'):
                trials = Trials()
                best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

            # Step 3: Create and train the best model with the best hyperparameters
            best_model = None
            if best_model_name == "Random Forest":
                best_model = RandomForestClassifier(**best_params)
            elif best_model_name == "XGBClassifier":
                best_model = XGBClassifier(**best_params)
            elif best_model_name == "CatBoost Classifier":
                best_model = CatBoostClassifier(**best_params, verbose=False)
            elif best_model_name == "Gradient Boosting":
                best_model = GradientBoostingClassifier(**best_params)
            elif best_model_name == "AdaBoost Classifier":
                best_model = AdaBoostClassifier(**best_params)
            else:
                raise ValueError(f"Model {best_model_name} not supported for hyperparameter tuning.")

            best_model.fit(X_train, y_train)

            # Re-evaluate the tuned model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Set the run name to "Best Model: <model_name>"
            run_name = f"Best Model: {best_model_name}"

            # Log the best model's performance in MLflow with a custom run name
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(best_params)
                mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

            # Save the best model to a file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)


def get_hyperparameter_space(model_name):
    """
    Return the hyperparameter search space for the given model.
    """
    if model_name == "Random Forest":
        return {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_depth': hp.choice('max_depth', [3, 5, 10, None]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
            'max_features': hp.choice('max_features', [None, 'sqrt', 'log2'])
        }
    elif model_name == "XGBClassifier":
        return {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_depth': hp.choice('max_depth', [3, 5, 10]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0)
        }
    elif model_name == "CatBoost Classifier":
        return {
            'depth': hp.choice('depth', [4, 6, 10]),
            'iterations': hp.quniform('iterations', 100, 1000, 1),  # 100 to 1000 iterations
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
        }
    elif model_name == "Gradient Boosting":
        return {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_depth': hp.choice('max_depth', [3, 5, 10]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.7, 1.0)
        }
    elif model_name == "AdaBoost Classifier":
        return {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 1.0)
        }
    else:
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning.")
