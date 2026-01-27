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

    def drop_cols(self, df: pd.DataFrame):
        try:
            # Only drop columns that actually exist in the dataframe
            cols_to_drop = [col for col in self._schema_config['drop_columns'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"we have sucesfully removed {cols_to_drop}")
            else:
                logger.info("No columns to drop - they were already transformed by OHE")
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    def pipeline_ohe(self, df: pd.DataFrame):
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            logger.info("correctly one hot encoder object created")
            col_transformer = ColumnTransformer([
                ('ohe', ohe, self._schema_config["ohe_columns"])
            ], remainder='passthrough')

            ohe_pipeline= Pipeline(steps=[
                ('column transformation', col_transformer)
            ])
            logger.info('created ohe pipeline with sklearn')

            return ohe_pipeline
        
        except Exception as e:
            logger.info(f"Error while creating sklearn pipeline")
            raise MyException(e,sys)
        
    def pipeline_scaler(self, df:pd.DataFrame):
        try:
            scaler = StandardScaler()
            logger.info("correctly scaler object created")
            
            columns = df.columns

            col_transformer = ColumnTransformer([
                ('standard_scaling', scaler,columns)
                
            ], remainder='passthrough')

            scaler_pipeline= Pipeline(steps=[
                ('column transformation', col_transformer)
            ])
            logger.info('created scaler pipeline with sklearn')

            return scaler_pipeline
        
        except Exception as e:
            logger.info(f"Error while creating sklearn pipeline")
            raise MyException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            if not len(self.data_validation_artifact.message) == 0:
                raise Exception(self.data_validation_artifact.message)
            logger.info('we are stating to transform our data')
            logger.info('we are reading data from our files')

            train_df = Data_Transformer.read_csv(self.data_injestion_artifact.trained_file_path)
            test_df = Data_Transformer.read_csv(self.data_injestion_artifact.test_file_path)

            # Apply transformations in correct order
            train_df = self.pdays_transformation(df=train_df)
            train_df = self.map_target(df=train_df)
            
            test_df = self.pdays_transformation(df=test_df)
            test_df = self.map_target(df=test_df)

            # Separate target BEFORE OHE
            y_train_df = train_df['Target']
            y_test_df = test_df['Target']
            
            # Remove Target column before OHE pipeline
            train_df_no_target = train_df.drop(columns=['Target'])
            test_df_no_target = test_df.drop(columns=['Target'])
            
            # Now apply OHE pipeline (BEFORE dropping columns)
            ohe_pipeline = self.pipeline_ohe(df=train_df_no_target)
            ohe_pipeline.fit(train_df_no_target)
            logger.info("Initializing transformation for Training-data")
            train_transformed = ohe_pipeline.transform(train_df_no_target)
            logger.info("Initializing transformation for Testing-data")
            test_transformed = ohe_pipeline.transform(test_df_no_target)

            # Get feature names from OHE pipeline
            ohe_transformer = ohe_pipeline.named_steps['column transformation'].transformers_[0][1]
            ohe_feature_names = ohe_transformer.get_feature_names_out(self._schema_config["ohe_columns"])
            
            # Get remaining columns (not in ohe_columns and not Target)
            remaining_cols = [col for col in train_df_no_target.columns if col not in self._schema_config["ohe_columns"]]
            all_feature_names = list(ohe_feature_names) + remaining_cols
            
            # Convert back to DataFrame with proper column names
            train_df = pd.DataFrame(train_transformed, columns=all_feature_names)
            test_df = pd.DataFrame(test_transformed, columns=all_feature_names)

            # Now drop columns AFTER OHE with preserved column names
            train_df = self.drop_cols(df=train_df)
            test_df = self.drop_cols(df=test_df)

            x_train_df = train_df
            x_test_df = test_df

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

            save_object(self.data_transformation_config.ohe_object_file_path, ohe_pipeline)
            save_object(self.data_transformation_config.scaling_object_file_path, preprocessor_pipeline)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logger.info("Saving transformation object and transformed files.")
            logger.info("Data transformation completed successfully")

            data_transformation_artifact = DataTransformationArtifact(
                tranformed_ohe_object_file_path=self.data_transformation_config.ohe_object_file_path,
                transformed_scaling_object_file_path=self.data_transformation_config.scaling_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)

            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)