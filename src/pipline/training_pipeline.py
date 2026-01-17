import sys
from src.entity.config_entity import (DataIngestionConfig)
from src.entity.artifact_entity import (DataIngestionArtifact)
from  src.logger import logger
from src.components.data_ingestion import Datainjestion
from src.exception import MyException

class Training_pipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_injestion(self)-> DataIngestionArtifact:
        try:
            logger.info("Entered the start_data_ingestion method of TrainPipeline class")
            logger.info("Getting the data from mongodb")
            dataInjestion = Datainjestion(datainjestion_config=self.data_ingestion_config)
            data_artifact = dataInjestion.initiate_data_injestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def run_pipeline(self):
        try:
            data_artifact = self.start_data_injestion()

            return data_artifact
        
        except Exception as e:
            raise MyException(e,sys)