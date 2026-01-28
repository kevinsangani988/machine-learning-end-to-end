import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logger
from pandas import DataFrame

import sys
from src.exception import MyException


class BankData:
    def __init__(
        self,
        age,
        job,
        marital,
        education,
        default,
        balance,
        housing,
        loan,
        contact,
        day,
        month,
        duration,
        campaign,
        pdays,
        previous,
        poutcome
    ):
        """
        Bank Data constructor
        Input: all features required by the trained model for prediction
        """
        try:
            # Numeric features
            self.age = age
            self.balance = balance
            self.day = day
            self.duration = duration
            self.campaign = campaign
            self.pdays = pdays
            self.previous = previous

            # Categorical features
            self.job = job
            self.marital = marital
            self.education = education
            self.default = default
            self.housing = housing
            self.loan = loan
            self.contact = contact
            self.month = month
            self.poutcome = poutcome
            self.pdays_unknown = 1 if int(self.pdays) == -1 else 0

        except Exception as e:
            raise MyException(e, sys) from e
    
    def get_data_as_dict(self):

        try:
            input_data = {
                "age" : [self.age],
                "balance" : [self.balance],
                "day" : [self.day],
                "duration" : [self.duration],
                "campaign" : [self.campaign],
                "pdays" : [self.pdays],
                "pdays_unknown": [self.pdays_unknown],
                "previous" : [self.previous],
                "job" : [self.job],
                "marital" : [self.marital],
                "education" : [self.education],
                "default" : [self.default],
                "housing" : [self.housing],
                "loan" : [self.loan],
                "contact" : [self.contact],
                "month" : [self.month],
                "poutcome" : [self.poutcome]

            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_data_as_dataframe(self,) -> DataFrame:

        try:
            input_data_dict = self.get_data_as_dict()
            # Remove pdays_unknown - it will be created by pdays_transformation
            if 'pdays_unknown' in input_data_dict:
                del input_data_dict['pdays_unknown']
            input_df = DataFrame(input_data_dict)
            return input_df
        except Exception as e:
            raise MyException(e,sys)
        
    def predict(self, dataframe: DataFrame, prediction_pipeline_config: VehiclePredictorConfig= VehiclePredictorConfig() ):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

            model = Proj1Estimator(bucket_name=prediction_pipeline_config.model_bucket_name,
                                   model_path=prediction_pipeline_config.model_file_path)
            prediction = model.predict(dataframe=dataframe)

            return prediction
        except Exception as e:
            raise MyException(e,sys)

        