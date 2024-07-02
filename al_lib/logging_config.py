import logging
from logging.handlers import RotatingFileHandler

# Configure the basic logger
# Specify the date - format of the log file


def create_logger(name="root", log_file_path="log_file.txt"):
    # Create a custom logger
    logger = logging.getLogger(name)
    # Set logging level for our logger
    logger.setLevel(logging.INFO)

    # Add the handlers (file and console)
    file_handler = RotatingFileHandler(
        filename=log_file_path,  # accepts also the path to the file since v3.6
        maxBytes=200000,
        backupCount=5,
    )
    console_handler = logging.StreamHandler()
    # console = logging.StreamHandler(sys.stdout)

    # Define the format
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt=time_format
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger if not already added
    if logger.handlers == []:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
