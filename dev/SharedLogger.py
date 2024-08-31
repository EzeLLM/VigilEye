import logging
import os

class CustomLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        log_filename = 'application.log'
        
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add file handler to logger
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

# Usage:
# from custom_logger import CustomLogger
# logger = CustomLogger().get_logger()
# logger.info('This is an info message')
