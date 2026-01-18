import os
import sys

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import MyException
from src.data_access.proj1_data import Proj1Data
import pandas as pd
from src.logger import logger
from src.constants import DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

from sklearn.model_selection import train_test_split


class Datainjestion:

    def __init__(self, datainjestion_config : DataIngestionConfig = DataIngestionConfig()):

        try:
            self.datainjestion_config = datainjestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    
    def export_data_into_feature_store(self):

        try:
            logger.info(f"Exporting data from mongodb")
            data = Proj1Data()
            df = data.fetch_data(collection_name=self.datainjestion_config.collection_name)
            logger.info(f"Shape of dataframe: {df.shape}")
            file_path = self.datainjestion_config.feature_store_file_path
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Saving exported data into feature store file path: {file_path}")
            df.to_csv(file_path, index=False,header=True)

            return df

        except Exception as e:
            raise 
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame ) -> None:

        try:
            train_set,test_set = train_test_split(dataframe, test_size=self.datainjestion_config.train_test_split_ratio)
            logger.info("Performed train test split on the dataframe")
            logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            file_path = self.datainjestion_config.training_file_path
            directory = os.path.dirname(file_path)
            os.makedirs(directory,exist_ok=True)
            logger.info(f"Exporting train and test file path.")

            train_set.to_csv(self.datainjestion_config.training_file_path,index=False,header=True )
            test_set.to_csv(self.datainjestion_config.testing_file_path,index=False,header=True )

            logger.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise MyException(e,sys)
        
    
    def initiate_data_injestion(self) -> DataIngestionArtifact: 
        logger.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            
            df = self.export_data_into_feature_store()

            logger.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe=df)


            logger.info("Performed train test split on the dataset")

            logger.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            dataInjestion_artifact = DataIngestionArtifact(trained_file_path=self.datainjestion_config.training_file_path,test_file_path= self.datainjestion_config.testing_file_path)
            
        
            
            logger.info(f"Data ingestion artifact: {dataInjestion_artifact}")

            return dataInjestion_artifact
        
        except Exception as e:
            raise MyException(e,sys)