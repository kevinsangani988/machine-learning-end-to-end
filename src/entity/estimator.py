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

            # Step 1: Apply OHE transformations using the pre-trained OHE object
            logging.info("Applying OHE transformations")
            transformed_feature = self.ohe_object.transform(dataframe)
            
            # Convert to DataFrame with proper column names if needed
            if isinstance(transformed_feature, np.ndarray):
                # Get column names from the OHE pipeline's ColumnTransformer
                try:
                    # Get OHE feature names
                    ohe_transformer = self.ohe_object.named_steps['column transformation'].transformers_[0][1]
                    ohe_feature_names = list(ohe_transformer.get_feature_names_out())
                    
                    # Get numeric columns that passed through (not OHE'd)
                    # These are columns from input dataframe that aren't in the OHE list
                    ohe_columns = set(['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome'])
                    remaining_cols = [col for col in dataframe.columns if col not in ohe_columns]
                    
                    # Combine all feature names
                    all_feature_names = ohe_feature_names + remaining_cols
                    transformed_feature = pd.DataFrame(transformed_feature, columns=all_feature_names)
                except:
                    # Fallback if unable to extract column names
                    transformed_feature = pd.DataFrame(transformed_feature)

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