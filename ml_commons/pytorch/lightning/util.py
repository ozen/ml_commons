import logging


def override_lightning_logger():
    from ml_commons.util.logger import get_default_handler
    logger = logging.getLogger('lightning')
    logger.handlers = []
    handler = get_default_handler()
    logger.addHandler(handler)
    logger.propagate = False
