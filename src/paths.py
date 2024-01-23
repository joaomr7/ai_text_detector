import os
from pathlib import Path

DATA_PATH = Path('data')
LOGS_PATH = os.path.join(os.getcwd(), 'logs')
ARTIFACTS_PATH = Path('artifacts')

def crate_directory_if_not_exist(directory_path):
    '''
    This function create the given directory if it do not exists
    '''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)