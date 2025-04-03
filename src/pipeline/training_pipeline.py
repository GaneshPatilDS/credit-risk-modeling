import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Define the stage names for better logging
STAGE_NAME_INGESTION = "Data Ingestion"
STAGE_NAME_TRANSFORMATION = "Data Transformation"
STAGE_NAME_TRAINING = "Model Training"

if __name__ == '__main__':
    try:
        # Stage 1: Data Ingestion
        logging.info(f"\n{'*' * 30}")
        logging.info(f">>>>>> Stage: {STAGE_NAME_INGESTION} started <<<<<<")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f">>>>>> Stage: {STAGE_NAME_INGESTION} completed <<<<<<\n{'x' * 100}\n")
        
        # Stage 2: Data Transformation
        logging.info(f"{'*' * 30}")
        logging.info(f">>>>>> Stage: {STAGE_NAME_TRANSFORMATION} started <<<<<<")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f">>>>>> Stage: {STAGE_NAME_TRANSFORMATION} completed <<<<<<\n{'x' * 100}\n")
        
        # Stage 3: Model Training
        logging.info(f"{'*' * 30}")
        logging.info(f">>>>>> Stage: {STAGE_NAME_TRAINING} started <<<<<<")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f">>>>>> Stage: {STAGE_NAME_TRAINING} completed <<<<<<\n{'x' * 100}\n")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise CustomException(e, sys)

  