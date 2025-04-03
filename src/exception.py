import sys
from src.logger import logging

# Function to format error details into a readable message
def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in {file_name} at line {exc_tb.tb_lineno}: {str(error)}"

# Custom exception class to handle errors with detailed messages
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    
    
    
    
    
    
    