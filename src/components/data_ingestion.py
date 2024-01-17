import os
from src.paths import ARTIFACTS_PATH, DATA_PATH
from src.paths import crate_directory_if_not_exist
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_PATH, 'train.csv')
    test_data_path: str = os.path.join(ARTIFACTS_PATH, 'test.csv')
    raw_data_path: str = os.path.join(ARTIFACTS_PATH, 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion.')

        try:
            df = pd.read_csv(DATA_PATH + '\Training_Essay_Data.csv')
            logging.info('Read the dataset as dataframe.')

            crate_directory_if_not_exist(ARTIFACTS_PATH)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-Test split initiated.')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['generated'])

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