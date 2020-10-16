
def override_lightning_logger():
    import logging
    from ml_commons.util.logger import get_default_handler
    logger = logging.getLogger('lightning')
    logger.handlers = []
    handler = get_default_handler()
    logger.addHandler(handler)
    logger.propagate = False


def import_module_attr(name):
    from importlib import import_module
    name_parts = name.split('.')
    module = import_module('.'.join(name_parts[:-1]))
    cls = getattr(module, name_parts[-1])
    return cls
