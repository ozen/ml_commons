import copy
import logging
from pytorch_lightning import Trainer


def add_trainer_args(source, destination=None, allowed=None):
    """
    Add Trainer init arguments from source to destination if the arg name is in the allowed list.
    If an allowed arg is not in the source, use the default value instead.
    :param source: Source dictionary
    :param destination: Destination dictionary. If None, a new dictionary is created
    :param allowed: list of names of allowed args
    :return: destination dictionary
    """
    if destination is None:
        destination = {}

    depr_arg_names = Trainer.get_deprecated_arg_names()

    for arg, arg_types, arg_default in Trainer.get_init_arguments_and_types():
        if arg not in depr_arg_names and (allowed is None or arg in allowed):
            destination[arg] = source[arg] if arg in source else arg_default

    return destination


def update_config_with_hparams(config, hparams):
    cfg = copy.deepcopy(config)
    if 'hparams' not in cfg:
        cfg['hparams'] = hparams
    else:
        cfg['hparams'].update(hparams)
    return cfg


def generate_random_hparams(parameters):
    hparams = {}
    for param in parameters:
        if param['type'] == 'range':
            hparams[param['name']] = param['bounds'][0]
        elif param['type'] == 'choice':
            hparams[param['name']] = param['values'][0]
    return hparams


def override_lightning_logger():
    from ml_commons.util.logger import get_default_handler
    logger = logging.getLogger('lightning')
    logger.handlers = []
    handler = get_default_handler()
    logger.addHandler(handler)
    logger.propagate = False
