import os
import sys
import inspect
import shutil
from glob import glob
from importlib import import_module
from argparse import ArgumentParser, FileType

from ml_commons.pytorch.lightning import AxLightningModule
from ml_commons.pytorch.lightning.config import get_cfg, AxCfgNode
from ml_commons.pytorch.lightning.util import override_lightning_logger
from ml_commons.util.stdout_capturing import CaptureStdout


def setup_experiment(config_file=None):
    """
    Sets up the experiment, returns experiment's AxLightningModule and configuration object.
    """
    # override lightning logger to match its style with ours
    override_lightning_logger()

    if config_file is None:
        # parse config_file input argument
        parser = ArgumentParser()
        parser.add_argument('config_file', type=FileType('r'))
        args = parser.parse_args()
        config_file = args.config_file.name

    # create the config dictionary
    cfg: AxCfgNode = get_cfg(config_file)

    # add source path to the sys.path for module import
    sys.path.append(cfg.source_path)

    # import experiment module
    module_parts = cfg.module.split('.')
    module = import_module('.'.join(module_parts[:-1]))
    cls: AxLightningModule = getattr(module, module_parts[-1])

    # merge model config
    if hasattr(cls, 'get_model_cfg'):
        model_cfg: AxCfgNode = cls.get_model_cfg()
        model_cfg.merge_from_other_cfg(cfg.model)
        cfg.defrost()
        cfg.model = model_cfg
        cfg.freeze()

    # create experiment directory
    os.makedirs(cfg.experiment_path)

    # copy config file
    shutil.copy(config_file, os.path.join(cfg.experiment_path, os.path.basename(config_file)))

    # copy sources
    module_path = inspect.getfile(module)
    module_dir = os.path.dirname(module_path)
    dest_dir = os.path.join(cfg.experiment_path, 'source')
    os.makedirs(dest_dir, exist_ok=True)
    for match_pattern in cfg.log_files:
        for source_file in glob(os.path.join(module_dir, match_pattern)):
            if os.path.isfile(source_file):
                os.makedirs(os.path.dirname(source_file), exist_ok=True)
                shutil.copy(source_file, os.path.join(dest_dir, os.path.basename(source_file)))

    return cls, cfg


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
        cls, cfg = setup_experiment()

        # run main function
        with CaptureStdout(os.path.join(cfg.experiment_path, 'cout.txt')):
            function(cls, cfg)

    return function


def optimize_and_train():
    cls, cfg = setup_experiment()
    with CaptureStdout(os.path.join(cfg.experiment_path, 'cout.txt')):
        cls.optimize_and_train(cfg)


def train():
    cls, cfg = setup_experiment()
    with CaptureStdout(os.path.join(cfg.experiment_path, 'cout.txt')):
        cls.fit(cfg, fast_dev_run=True)
