import logging


class Logger:
    def __init__(self):
        # if running on startup change to "/home/pi/code/LicencePlatesDetector/logs.txt"
        self.log_file_path = "logs.txt"
        self.formatter_string = '%(asctime)s [%(levelname)s]: %(message)s'

    def get_console_logger(self):
        return self.setup_logger("console_logger")

    def get_file_logger(self):
        return self.setup_logger("file_logger", log_file=self.log_file_path)

    def setup_logger(self, name="logger", log_file='', level=logging.INFO):
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter(self.formatter_string))
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger


console_logger = Logger().get_console_logger()
file_logger = Logger().get_file_logger()
