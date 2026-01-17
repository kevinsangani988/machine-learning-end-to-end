from src.logger import logging
from src.exception import MyException
import sys
from src.pipline.training_pipeline import Training_pipeline
# logger = create_logger()
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")

# try:
#     a = 1+'Z'
# except Exception as e:
#     logger.info(e)
#     raise MyException(e, sys) from e

pipeline = Training_pipeline()
pipeline.run_pipeline()