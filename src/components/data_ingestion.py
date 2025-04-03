import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.constant import * 
# Ensure this imports any required constants or column names

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('../../', 'artifacts', 'train.csv')
    test_data_path: str = os.path.join('../../', 'artifacts', 'test.csv')
    raw_data_path: str = os.path.join('../../', 'artifacts', 'raw.csv')
    data_file_path: str = Path('C:/Users/Harshali/Documents/CRM/notebooks/data/Data_file.xlsx')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            if self.ingestion_config.data_file_path.exists():
                df = pd.read_excel(self.ingestion_config.data_file_path, names=column_names)
                logging.info('Dataset read as pandas DataFrame')
            else:
                raise FileNotFoundError(f"File not found: {self.ingestion_config.data_file_path}")

            # Check and log NaN values
            nan_percentage = df.isna().mean() * 100
            logging.info('NaN values percentage per column:')
            logging.info(nan_percentage)

            duplicate_rows = df.duplicated().sum()
            logging.info(f'Found {duplicate_rows} duplicate rows')
            df.drop_duplicates(keep='first', inplace=True)
            logging.info('Dropped duplicates')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved')

            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)


