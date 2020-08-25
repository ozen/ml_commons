import inspect
import os
import shutil
import sys
from glob import glob
from importlib import import_module

from ml_commons.pytorch.lightning.cfgnode import LightningCfgNode
from ml_commons.pytorch.lightning.config import get_cfg
from ml_commons.pytorch.lightning.util import override_lightning_logger


def import_modules(cfg: LightningCfgNode):
    # import model module
    module_parts = cfg.module.split('.')
    model_module = import_module('.'.join(module_parts[:-1]))
    model_module_class = getattr(model_module, module_parts[-1])

    # import data module
    module_parts = cfg.data_module.split('.')
    data_module = import_module('.'.join(module_parts[:-1]))
    data_module_class = getattr(data_module, module_parts[-1])

    return model_module_class, data_module_class


def setup_experiment(config_file=None):
    """
    Sets up the experiment, returns experiment's LightningModule and configuration object.
    """
    # override lightning logger to match its style with ours
    override_lightning_logger()

    # create the config dictionary
    cfg: LightningCfgNode = get_cfg(config_file)

    # add source path to the sys.path for module import
    sys.path.append(cfg.source_path)

    # import model and data modules
    model_module_class, data_module_class = import_modules(cfg)

    # merge model config
    cfg.merge_get_cfg('model', model_module_class)

    # merge data config
    cfg.merge_get_cfg('data', data_module_class)

    # create experiment directory
    os.makedirs(cfg.experiment_path)

    # copy config file
    shutil.copy(cfg.config_file_path, os.path.join(cfg.experiment_path, os.path.basename(cfg.config_file_path)))

    # copy sources
    model_module = import_module('.'.join(cfg.module.split('.')[:-1]))
    module_path = inspect.getfile(model_module)
    module_dir = os.path.dirname(module_path)
    dest_dir = os.path.join(cfg.experiment_path, 'source')
    os.makedirs(dest_dir, exist_ok=True)
    for match_pattern in cfg.log_files:
        for source_file in glob(os.path.join(module_dir, match_pattern)):
            if os.path.isfile(source_file):
                os.makedirs(os.path.dirname(source_file), exist_ok=True)
                shutil.copy(source_file, os.path.join(dest_dir, os.path.basename(source_file)))

    return model_module_class, data_module_class, cfg
