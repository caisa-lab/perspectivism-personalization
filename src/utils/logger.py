import logging.config
import logging
import yaml
import sys

with open('configs/logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)

def get_logger(name: str):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger(name)

    fileHandler = logging.FileHandler("../logs/debug.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger