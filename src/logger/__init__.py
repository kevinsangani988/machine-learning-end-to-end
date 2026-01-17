import logging
import os
from datetime import datetime
import time

def create_logger():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    dir_name = 'logging_details'

    os.makedirs(dir_name, exist_ok=True)


    file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    file_path = os.join(ROOT_DIR, file_name)


    logger = logging.getLogger('Base logger')
    logger.setLevel(logging.debug)

    formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(message)s -- %(name)s')

    filehandler = logging.FileHandler(file_path)
    filehandler.setFormatter(formatter)
    filehandler.setLevel(logging.debug)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.debug)

    logger.addHandler(filehandler)
    logger.addHandler(console_handler)

create_logger(__file__)
