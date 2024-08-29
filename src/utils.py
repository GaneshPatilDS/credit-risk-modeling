import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

# Function to save an object using pickle
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Function to load an object using pickle
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
# Function to evaluate models and return a report
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            report[name] = {
                'Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred, average='weighted'),
                'Recall': recall_score(y_test, y_test_pred, average='weighted'),
                'F1': f1_score(y_test, y_test_pred, average='weighted')
            }
        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

class CustomOrdinalEncoder:
    def fit(self, X, y=None):
        self.education_labels = {
            'SSC': 1,
            '12TH': 2,
            'GRADUATE': 3,
            'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4,
            'OTHERS': 1,
            'PROFESSIONAL': 3
        }
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.apply(lambda x: x.map(self.education_labels))
        elif isinstance(X, np.ndarray):
            X = np.vectorize(self.education_labels.get)(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)    