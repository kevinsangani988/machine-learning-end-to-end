import sys
from src.entity.config_entity import *
from src.entity.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact)
from  src.logger import logger
from src.components.data_ingestion import Datainjestion
from src.components.data_validation import Datavalidator
from src.components.model_trainer import Model_Train
from src.components.model_evaluation import ModelEvaluation
from src.components.data_transformation import Data_Transformer
from src.components.model_pusher import ModelPusher
from src.exception import MyException

class Training_pipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DatavalidationCOnfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()


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
        
    def start_data_transformation(self, data_injestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation = Data_Transformer(data_injestion_artifact=data_injestion_artifact,
                                                  data_validation_artifact=data_validation_artifact,
                                                  data_transformation_config=DataTransformationConfig)
            data_transformation_Artifact = data_transformation.initiate_data_transformation()
            logger.info("data transformation artifact done")
            return data_transformation_Artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_model_training(self,
                              data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer = Model_Train(model_trainer_config=ModelTrainerConfig,
                                        data_transformation_artifact=data_transformation_artifact)
            model_training_artifact = model_trainer.initiate_model_training()
            logger.info('model training successful')
            return model_training_artifact
        except Exception as e:
            raise MyException(e,sys)

    def start_model_evaluation(self, data_injestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact,
                               data_transformation_artifact : DataTransformationArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_injestion_artifact=data_injestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact,
                                               data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys)

        
    def run_pipeline(self):
        try:
            data_injestion_artifact = self.start_data_injestion()
            data_validation_artifact = self.start_data_validation( data_ingestion_artifact=data_injestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_injestion_artifact=data_injestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            model_training_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_injestion_artifact=data_injestion_artifact,
                                                                    model_trainer_artifact=model_training_artifact, 
                                                                    data_transformation_artifact=data_transformation_artifact)

            if not model_evaluation_artifact.is_model_accepted:
                logger.info(f"Model not accepted.")
                return None
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        
        except Exception as e:
            raise MyException(e,sys)