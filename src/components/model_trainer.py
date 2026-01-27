import sys
from sklearn.ensemble import RandomForestClassifier
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logger
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.estimator import MyModel



class Model_Train:

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise MyException(e,sys)
        
    def model(self,test_arr:np.array, train_arr:np.array):

        try:
            x_train,x_test,y_train,y_test = train_arr[:,:-1], test_arr[:,:-1], train_arr[:,-1],test_arr[:,-1]
            model = RandomForestClassifier(n_estimators=self.model_trainer_config._n_estimators,
                                           max_depth=self.model_trainer_config._max_depth)
            logger.info('starting model training')
            model.fit(x_train,y_train)
            logger.info('model is ready for prediction')
            y_test_pred = model.predict(x_test)

            precision = precision_score(y_test,y_test_pred)
            f1_score_ = f1_score(y_test,y_test_pred)
            recall = recall_score(y_test, y_test_pred)

            metrics = ClassificationMetricArtifact(f1_score=f1_score_, precision_score=precision,recall_score=recall)

            return model, metrics
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_training(self):
        try:
            train_array = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            model,metrics = self.model(test_arr=test_array,train_arr=train_array)

            ohe_obj = load_object(file_path=self.data_transformation_artifact.tranformed_ohe_object_file_path)
            scaling_obj = load_object(file_path=self.data_transformation_artifact.transformed_scaling_object_file_path)
            logger.info("OHE and scaling objects loaded.")
            logger.debug("Saving new model as performace is better than previous one.")
            my_model = MyModel(ohe_object=ohe_obj, scaling_object=scaling_obj, trained_model_object=model)
            logger.info("Saved final model object that includes both preprocessing and the trained model")
            save_object(self.model_trainer_config.trained_model_file_path, my_model)

            model_trainer_artifact= ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                         metric_artifact=metrics)
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)