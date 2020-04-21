import logging
from tqdm import tqdm


class TqdmStream(object):
    @classmethod
    def write(cls, msg):
        if msg == '\n':
            return
        tqdm.write(msg, end='\n')


def get_logger(name: str = 'ml_commons') -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.level == 0:
        logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = get_default_handler()
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_default_handler():
    handler = logging.StreamHandler(stream=TqdmStream)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler
