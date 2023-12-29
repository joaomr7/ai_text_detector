import logging
import os
from datetime import datetime

from src.paths import LOGS_PATH, crate_directory_if_not_exist

LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

crate_directory_if_not_exist(LOGS_PATH)

LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(filename)s - %(levelname)s - %(message)s',
    datefmt='%m-%d-%Y %I:%M:%S %p',
    level=logging.INFO
)