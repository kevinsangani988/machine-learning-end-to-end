import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, ohe_object: Pipeline, scaling_object: Pipeline, trained_model_object: object):
        """
        :param ohe_object: One-hot encoder pipeline object
        :param scaling_object: Scaling/preprocessing pipeline object
        :param trained_model_object: Input Object of trained model 
        """
        self.ohe_object = ohe_object
        self.scaling_object = scaling_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts raw input dataframe, applies OHE and scaling transformations,
        and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")
            
            # Step 0: Apply pdays_transformation if not already done
            # Create pdays_unknown column from pdays
            if 'pdays_unknown' not in dataframe.columns and 'pdays' in dataframe.columns:
                logging.info("Applying pdays_transformation")
                dataframe = dataframe.copy()
                dataframe['pdays_unknown'] = dataframe['pdays'].apply(lambda x: 0 if x == -1 else x)

            # Step 1: Apply OHE transformations using the pre-trained OHE object
            logging.info("Applying OHE transformations")
            transformed_feature = self.ohe_object.transform(dataframe)
            
            # Convert to DataFrame with proper column names if needed
            if isinstance(transformed_feature, np.ndarray):
                # Get column names from the OHE pipeline's ColumnTransformer
                try:
                    # Get all feature names from the OHE pipeline
                    ct = self.ohe_object.named_steps['column transformation']
                    all_feature_names = list(ct.get_feature_names_out())
                    transformed_feature = pd.DataFrame(transformed_feature, columns=all_feature_names)
                    logging.info(f"OHE DataFrame created with {len(all_feature_names)} columns")
                except Exception as e:
                    logging.warning(f"Could not extract column names from OHE: {e}. Creating unnamed DataFrame.")
                    transformed_feature = pd.DataFrame(transformed_feature)

            # Step 1.5: Drop columns that should not be passed to scaler
            # These columns were dropped during training before fitting the scaler
            drop_columns_list = ['contact', 'day', 'month', 'pdays']
            cols_to_drop = [col for col in drop_columns_list if col in transformed_feature.columns]
            if cols_to_drop:
                logging.info(f"Dropping columns before scaling: {cols_to_drop}")
                transformed_feature = transformed_feature.drop(columns=cols_to_drop)
            
            # Ensure it's still a DataFrame with column names
            if not isinstance(transformed_feature, pd.DataFrame):
                logging.error("transformed_feature is not a DataFrame after drop operation")
                raise ValueError("Transformation resulted in non-DataFrame object")
                
            logging.info(f"Before scaling: shape={transformed_feature.shape}, columns count={len(transformed_feature.columns)}")
            logging.info(f"Column names: {list(transformed_feature.columns)}")

            # Step 2: Apply scaling transformations using the pre-trained scaling object
            logging.info("Applying scaling transformations")
            transformed_feature = self.scaling_object.transform(transformed_feature)

            # Step 3: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"