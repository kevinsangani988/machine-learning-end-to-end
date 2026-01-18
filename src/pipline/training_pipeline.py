import sys
from src.entity.config_entity import (DataIngestionConfig, DatavalidationCOnfig)
from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact)
from  src.logger import logger
from src.components.data_ingestion import Datainjestion
from src.components.data_validation import Datavalidator
from src.exception import MyException

class Training_pipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_injestion(self)-> DataIngestionArtifact:
        try:
            logger.info("Entered the start_data_ingestion method of TrainPipeline class")
            logger.info("Getting the data from mongodb")
            dataInjestion = Datainjestion(datainjestion_config=self.data_ingestion_config)
            data_injestion_artifact = dataInjestion.initiate_data_injestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_injestion_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = Datavalidator(data_injestion_artifact=data_ingestion_artifact, data_validator_config=DatavalidationCOnfig())
            data_validation_artifact = data_validation.initialize_data_validation()
            logger.info("Performed the data validation operation")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def run_pipeline(self):
        try:
            data_injestion_artifact = self.start_data_injestion()
            data_validation_artifact = self.start_data_validation( data_ingestion_artifact=data_injestion_artifact)

            return data_injestion_artifact, data_validation_artifact
        
        except Exception as e:
            raise MyException(e,sys)