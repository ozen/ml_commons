import os
import sys
import inspect
import shutil
from glob import glob
from importlib import import_module

from ml_commons.pytorch.lightning.config import get_cfg, LightningCfgNode
from ml_commons.pytorch.lightning.experimenter import Experimenter
from ml_commons.pytorch.lightning.util import override_lightning_logger
from ml_commons.util.stdout_capturing import CaptureStdout


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
    model_module_class, data_module_class = import_module(cfg)

    # merge model config
    cfg.merge_get_cfg('model', model_module_class)

    # merge data config
    cfg.merge_get_cfg('data', data_module_class)

    # create experiment directory
    os.makedirs(cfg.experiment_path)

    # copy config file
    shutil.copy(config_file, os.path.join(cfg.experiment_path, os.path.basename(config_file)))

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


def automain(function):
    """
    Sets up the experiment and automatically runs the wrapped function
    """
    if function.__module__ == "__main__":
        # ensure that automain is not used in interactive mode.
        main_filename = inspect.getfile(function)
        if main_filename == "<stdin>" or (
                main_filename.startswith("<ipython-input-")
                and main_filename.endswith(">")
        ):
            raise RuntimeError("Cannot use @automain decorator in interactive mode.")

        # setup the experiment
        model_class, data_class, cfg = setup_experiment()

        # run main function
        with CaptureStdout(os.path.join(cfg.experiment_path, 'stdout.txt')):
            function(model_class, data_class, cfg)

    return function


def optimize_and_fit():
    experimenter = Experimenter()
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.optimize_and_fit()


def fit():
    experimenter = Experimenter()
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.fit(fast_dev_run=True)


def resume():
    from argparse import ArgumentParser
    from argparse import FileType

    parser = ArgumentParser()
    parser.add_argument('config_file', type=FileType('r'))
    parser.add_argument('checkpoint_file', type=FileType('r'))
    args = parser.parse_args()
    config_file_path = args.config_file.name
    checkpoint_file_path = args.checkpoint_file.name

    experimenter = Experimenter(config_file_path)
    with CaptureStdout(os.path.join(experimenter.cfg.experiment_path, 'stdout.txt')):
        experimenter.resume(checkpoint_file_path)
