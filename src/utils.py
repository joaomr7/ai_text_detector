import os
import pickle

from src.exception import CustomException
from src.paths import crate_directory_if_not_exist

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        crate_directory_if_not_exist(dir_path)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    
    except Exception as e:
        raise CustomException(e)
