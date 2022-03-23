from logger import console_logger, file_logger


def run():
    console_logger.info("Program starts")
    file_logger.info("Program starts")

    file_logger.info("Program ends")
    console_logger.info("Program ends")


if __name__ == '__main__':
    run()
