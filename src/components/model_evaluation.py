from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logger
from src.utils.main_utils import load_object
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_injestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_injestion_artifact = data_injestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_best_model(self):
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)
            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def pdays_transformation(self, df: pd.DataFrame):
        try:
            df['pdays_unknown'] = df['pdays'].apply(lambda x: 0 if x==-1 else x )
            logger.info('transformed pdays column succesfully')
            return df
        except Exception as e:
            logger.error('unexpected error occured in pdays_transformation')
            raise MyException(e,sys)
    
    # def column_ohe(self, df: pd.DataFrame):
    #     try:
    #         ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    #         ohe_encoded = ohe.fit_transform(df[self._schema_config["ohe_columns"]])
    #         columns = ohe.get_feature_names_out()
    #         ohe_df = pd.DataFrame(ohe_encoded, columns=columns, index=df.index)
    #         df = pd.concat([df, ohe_df], axis=1)
    #         logger.info("correctly one hot encoded columns from our data")
    #         return df
    #     except Exception as e:
    #         raise MyException(e,sys)
    
    def map_target(self, df:pd.DataFrame):
        try:
            df['Target'] = df['Target'].map({'yes': 1, 'no': 0})
            logger.info("correctly mapped target column")
            return df
        except Exception as e:
            raise MyException(e,sys)

    def drop_cols(self,df:pd.DataFrame):
        try:
            # Only drop columns that actually exist in the dataframe
            cols_to_drop = [col for col in self._schema_config['drop_columns'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"we have sucesfully removed {cols_to_drop}")
            else:
                logger.info("No columns to drop - they were already removed by OHE pipeline")
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_injestion_artifact.test_file_path)

            logger.info("Test data loaded and now transforming it for prediction...")

            # Apply custom transformations
            x = test_df.copy()
            x = self.pdays_transformation(df=x)
            x = self.map_target(df=x)
            
            # Extract target
            y = x[TARGET_COLUMN]
            x_for_best_model = x.drop(columns=[TARGET_COLUMN], axis=1)
            
            # Prepare data for trained model (apply full transformations)
            x_main = x_for_best_model.copy()
            ohe_pipeline = load_object(self.data_transformation_artifact.tranformed_ohe_object_file_path)
            logger.info("OHE pipeline loaded/exists.")
            x_trained = ohe_pipeline.transform(x_main)
            
            # Convert to DataFrame with proper column names
            if isinstance(x_trained, np.ndarray):
                ohe_transformer = ohe_pipeline.named_steps['column transformation'].transformers_[0][1]
                ohe_feature_names = ohe_transformer.get_feature_names_out(self._schema_config["ohe_columns"])
                remaining_cols = [col for col in x_main.columns if col not in self._schema_config["ohe_columns"]]
                all_feature_names = list(ohe_feature_names) + remaining_cols
                x_trained = pd.DataFrame(x_trained, columns=all_feature_names, index=x_main.index)

            x_trained = self.drop_cols(df=x_trained)

            # Apply scaling for trained model
            scaler_pipeline = load_object(self.data_transformation_artifact.transformed_scaling_object_file_path)
            logger.info("Scaling pipeline loaded/exists.")
            x_trained = scaler_pipeline.transform(x_trained)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logger.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logger.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logger.info(f"Computing F1_Score for production model..")
                # Pass data with only custom transformations (pdays, target mapping) so best_model can apply OHE and scaler internally
                y_hat_best_model = best_model.predict(x_for_best_model)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logger.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logger.info(f"Result: {result}")
            return result
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logger.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e