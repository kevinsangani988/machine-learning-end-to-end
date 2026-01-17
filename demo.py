from src.logger import create_logger
from src.exception import MyException
import sys

logger = create_logger()
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")

try:
    a = 1+'Z'
except Exception as e:
    logger.info(e)
    raise MyException(e, sys) from e