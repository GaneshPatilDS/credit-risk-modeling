import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Preprocess the features
            data_scaled = preprocessor.transform(features)

            # Predict
            predictions = model.predict(data_scaled)
            return predictions

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)
