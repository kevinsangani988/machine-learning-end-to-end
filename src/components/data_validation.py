import sys 
from src.logger import logger
import pandas as pd
from src.entity.config_entity import DatavalidationCOnfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from src.exception import MyException
import os
import json

class Datavalidator():
    def __init__(self, data_injestion_artifact: DataIngestionArtifact , data_validator_config : DatavalidationCOnfig):
        try:
            self.data_injestion_artifact = data_injestion_artifact
            self.data_validator_config = data_validator_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    def column_validate(self,dataframe: pd.DataFrame):
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logger.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    def is_column_exist(self, df: pd.DataFrame):
        try:
            numerical_cols = []
            categorical_cols = []
            columns = df.columns

            for column in self._schema_config["categorical_columns"]:
                if column not in columns:
                    categorical_cols.append(column)
            
            if len(categorical_cols)>0:
                logger.info(f"Missing numerical column: {categorical_cols}")

            for column in self._schema_config["numerical_columns"]:
                if column not in columns:
                    numerical_cols.append(column)
            
            if len(numerical_cols)>0:
                logger.info(f"please give enough data here are missing cols{numerical_cols}")

            return False if len(categorical_cols)>0 or len(numerical_cols)>0 else True
        except Exception as e:
            raise MyException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def initialize_data_validation(self):
        try:
            validation_error_msg = ""
            logger.info("Starting data validation")
            train_df, test_df = (Datavalidator.read_data(file_path=self.data_injestion_artifact.trained_file_path),
                                 Datavalidator.read_data(file_path=self.data_injestion_artifact.test_file_path))
            
            status = self.column_validate(dataframe=train_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logger.info(f"All required columns present in testing dataframe: {status}")

            status = self.column_validate(dataframe=test_df)

            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logger.info(f"All required columns present in testing dataframe: {status}")

            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logger.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logger.info(f"All categorical/int columns present in testing dataframe: {status}")

            validation_status = len(validation_error_msg)==0

            data_validation_artifact = DataValidationArtifact(validation_status=validation_status,
                                                              message=validation_error_msg,
                                                              validation_report_file_path=self.data_validator_config.validation_report_file_path) 
            
            report_dir = os.path.dirname(self.data_validator_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            content = {
                "validation_status": validation_status,
                "message": validation_error_msg
            }

            with open(self.data_validator_config.validation_report_file_path, 'w') as f:
                json.dump(content, f)

            logger.info("Data validation artifact created and saved to JSON file.")
            logger.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e