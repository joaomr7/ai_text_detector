import os

LOGS_PATH = os.path.join(os.getcwd(), 'logs')

def crate_directory_if_not_exist(directory_path):
    '''
    This function create the given directory if it do not exists
    '''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)