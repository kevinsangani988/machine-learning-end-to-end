from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
import sys
import pandas as pd
import numpy as np
from src.constants import *
from src.exception import MyException
from src.utils.main_utils import read_yaml_file , save_object, save_numpy_array_data
from src.logger import logger
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Data_Transformer:

    def __init__(self, data_injestion_artifact : DataIngestionArtifact,
                 data_transformation_config : DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact ):
        self.data_injestion_artifact = data_injestion_artifact
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    @staticmethod
    def read_csv(file_path : str)->pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def pdays_transformation(self, df: pd.DataFrame):
        try:
            df['pdays_unknown'] = df['pdays'].apply(lambda x: 0 if x==-1 else x )
            logger.info('transformed pdays column succesfully')
            return df
        except Exception as e:
            logger.error('unexpected error occured in pdays_transformation')
            raise MyException(e,sys)
    
    def column_ohe(self, df: pd.DataFrame):
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ohe_encoded = ohe.fit_transform(df[self._schema_config["ohe_columns"]])
            columns = ohe.get_feature_names_out()
            ohe_df = pd.DataFrame(ohe_encoded, columns=columns, index=df.index)
            df = pd.concat([df, ohe_df], axis=1)
            logger.info("correctly one hot encoded columns from our data")
            return df
        except Exception as e:
            raise MyException(e,sys)
    
    def map_target(self, df:pd.DataFrame):
        try:
            df['Target'] = df['Target'].map({'yes': 1, 'no': 0})
            logger.info("correctly mapped target column")
            return df
        except Exception as e:
            raise MyException(e,sys)

    def drop_cols(self,df:pd.DataFrame):
        try:
            df = df.drop(columns=self._schema_config['drop_columns'])
            logger.info(f"we have sucesfully removed {self._schema_config['drop_columns']}")
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def pipeline_scaler(self, df:pd.DataFrame):
        try:
            scaler = StandardScaler()
            logger.info("correctly scaler object created")
            num_cols = df.columns
            col_transformer = ColumnTransformer([
                ('standard_scaling', scaler, num_cols)
            ], remainder='passthrough')

            final_pipeline= Pipeline(steps=[
                ('column transformation', col_transformer)
            ])
            logger.info('created final pipeline with sklearn')

            return final_pipeline
        except Exception as e:
            logger.info(f"Error while creating sklearn pipeline")
            raise MyException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            if not len(self.data_validation_artifact.message) == 0:
                raise Exception(self.data_validation_artifact.message)
            logger.info('we are stating to transform our data')
            logger.info('we are reading data from our files')

            train_df =Data_Transformer.read_csv(self.data_injestion_artifact.trained_file_path)
            test_df = Data_Transformer.read_csv(self.data_injestion_artifact.test_file_path)

            train_df = self.column_ohe(df=train_df)
            train_df = self.pdays_transformation(df=train_df)
            train_df = self.map_target(df=train_df)
            train_df = self.drop_cols(df=train_df)

            test_df = self.pdays_transformation(df=test_df)
            test_df = self.column_ohe(df=test_df)
            test_df = self.map_target(df=test_df)
            test_df = self.drop_cols(df=test_df)

            x_train_df = train_df.drop(columns=['Target'])
            y_train_df = train_df['Target']

            x_test_df = test_df.drop(columns=['Target'])
            y_test_df = test_df['Target']

            logger.info("Custom transformations applied to train and test data")

            logger.info("initiating column transformer pipeline")

            preprocessor_pipeline = self.pipeline_scaler(df=x_train_df)
            x_train_array = preprocessor_pipeline.fit_transform(x_train_df)
            logger.info("Initializing transformation for Testing-data")
            x_test_array = preprocessor_pipeline.transform(x_test_df)
            logger.info("Transformation done end to end to train-test df.")

            train_arr = np.c_[x_train_array, np.array(y_train_df)]
            test_arr = np.c_[x_test_array, np.array(y_test_df)]
            logger.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_pipeline)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logger.info("Saving transformation object and transformed files.")

            logger.info("Data transformation completed successfully")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)

            return data_transformation_artifact

        except Exception as e:
            return MyException(e,sys)