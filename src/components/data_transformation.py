import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import CustomOrdinalEncoder

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    """
    preprocessor_obj_file_path: str = os.path.join('../../', 'artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Class for data transformation.
    """

    def __init__(self):
        """
        Initialize the data transformation object.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Create and return the data transformation pipeline.
        """
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be numerical, ordinal-encoded, and one-hot encoded
            numerical_cols =  ['pct_tl_open_L6M','pct_tl_closed_L6M','Tot_TL_closed_L12M',
                               'pct_tl_open_L12M','pct_tl_closed_L12M','Tot_Missed_Pmnt',
                               'CC_TL','Home_TL','PL_TL','Secured_TL','Unsecured_TL',
                               'Other_TL','time_since_recent_payment','max_recent_level_of_deliq',
                               'num_deliq_6_12mts','num_times_60p_dpd','num_std_12mts','num_sub',
                               'num_sub_6mts','num_sub_12mts','num_dbt','num_dbt_12mts','num_lss',
                               'recent_level_of_deliq','CC_enq_L12m', 'PL_enq_L12m','time_since_recent_enq',
                               'enq_L3m','NETMONTHLYINCOME','Time_With_Curr_Empr','CC_Flag','PL_Flag',
                               'pct_PL_enq_L6m_of_ever','pct_CC_enq_L6m_of_ever','HL_Flag','GL_Flag']
            
            categorical_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
            ordinal_cols = ['EDUCATION']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('scaler', MinMaxScaler())
                ]
            )

            # Categorical Pipeline for nominal variables
            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder(sparse_output=False))
                ]
            )

            # Categorical Pipeline for ordinal variables
            ordinal_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', CustomOrdinalEncoder())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_cols)
            ])

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiate data transformation.
        """
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            # Define the ordering for Approved_Flag
            approved_flag_map = ['P1', 'P2', 'P3', 'P4']

            # Create a LabelEncoder object
            label_encoder = LabelEncoder()

            # Encode the 'Approved_Flag' column
            train_df['Approved_Flag'] = label_encoder.fit_transform(train_df['Approved_Flag'])
            test_df['Approved_Flag'] = label_encoder.transform(test_df['Approved_Flag'])

            logging.info('Converting output column into numerical form using Label Encoding')
            logging.info(f'Approved_Flag column converted to numerical values: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}')

            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            target_column_name = 'Approved_Flag'
            drop_columns = [target_column_name, 'Approved_Flag']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Train(input) Dataframe head: \n{input_feature_train_df.head(2).to_string()}')
            logging.info(f'Test(input) Dataframe head: \n{input_feature_test_df.head(2).to_string()}')

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformation_object()

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Log unique values of the target variable in train_arr
            unique_train_target_values = np.unique(train_arr[:, -1])
            logging.info(f'Unique values in the target variable of train_arr: {unique_train_target_values}')

            # Log unique values of the target variable in test_arr
            unique_test_target_values = np.unique(test_arr[:, -1])
            logging.info(f'Unique values in the target variable of test_arr: {unique_test_target_values}')


            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            # Save transformed data arrays as .npy files
            np.save('../../artifacts/transformed_train.npy', train_arr)
            np.save('../../artifacts/transformed_test.npy', test_arr)

            logging.info('Transformed training and testing data saved as .npy files')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info('Exception occurred in the initiate_data_transformation')
            raise CustomException(e, sys)
