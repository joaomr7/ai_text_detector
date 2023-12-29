import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    '''
    Function to return a string containing details about the error
    '''
    _, _, exc_tb = error_detail

    file_name = 'Unknown'
    line_number = 'Unknown'

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

    error_message = f'Error occured in python script name [{file_name}] line number [{line_number}] error message[{str(error)}]'

    return error_message

class CustomException(Exception):
    '''
    Custom exception class that process the error message with error_message_detail() and log the error
    '''
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, sys.exc_info())

        logging.error(self.error_message)

    def __str__(self):
        return self.error_message