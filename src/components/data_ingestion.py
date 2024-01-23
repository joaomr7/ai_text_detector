import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.paths import ARTIFACTS_PATH, DATA_PATH
from src.paths import crate_directory_if_not_exist
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    '''
    Configuration class for storing file path of the output files of the data ingestion component.

    Attributes
    ---
    * train_data_path: the file path to store the training data file.
    * test_data_path: the file path to store the test data file.
    * raw_data_path: the file path to store the raw data file.
    '''

    train_data_path: str = ARTIFACTS_PATH / 'train.csv'
    test_data_path: str = ARTIFACTS_PATH / 'test.csv'
    raw_data_path: str = ARTIFACTS_PATH / 'raw.csv'

class DataIngestion:
    '''
    Class responsible for data ingestion.
    '''

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def __clean_data(self, df: pd.DataFrame):
        '''
        Helper function to clean the given data frame.

        Parameters
        ---
        * df: data frame to clean.

        Return
        ---
        * the cleaned data frame.
        '''

        # remove invalid string
        df_clean = df[df['text'].str.contains('\w')]

        # remove invalid 'generated' values
        df_clean = df_clean[df_clean['generated'].isin([0, 1])]

        # remove text with less than 300 characters
        texts_lengths = np.array([len(text) for text in df_clean['text']])
        df_clean = df_clean[texts_lengths >= 300]

        return df_clean

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion.')

        try:
            df = pd.read_csv(DATA_PATH / 'Training_Essay_Data.csv')
            logging.info('Read the dataset as dataframe.')

            crate_directory_if_not_exist(ARTIFACTS_PATH)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data-cleaning initiated.')
            df_clean = self.__clean_data(df)

            logging.info('Train-Test split initiated.')
            train_set, test_set = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['generated'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed.')

        except Exception as e:
            raise CustomException(e)
        
        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )
    
def main():
    '''
    This function is responsible for running the data ingestion, data transformation and model training workflows
    '''

    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)

    model_trainer = ModelTrainer()
    train_metrics, test_metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print('Train metrics:', train_metrics)
    print('Test metrics:', test_metrics)

if __name__ == '__main__':
    main()