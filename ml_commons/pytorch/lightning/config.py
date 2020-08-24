import os
from argparse import ArgumentParser, FileType

from ml_commons.pytorch.lightning.cfgnode import LightningCfgNode
from ml_commons.util.path import timestamp_path


_C = LightningCfgNode()

# The root path. Defaults to the directory of config file.
_C.source_path = False

# Module path to the MLCModule, relative to source path
_C.module = 'module.Module'

# Module path to the MLCDataModule, relative to source path
_C.data_module = 'module.DataModule'

# Path to the experiments directory, relative to source path
_C.experiments_dir = 'experiments'

# Name of the experiment
_C.experiment_name = 'default'

# filename patterns relative to the module's directory to be saved to the logs. beware of loops.
_C.log_files = ['*']

# how many subprocesses to use for data loading
_C.num_workers = 0

# Batch size
_C.batch_size = 256

# Number of saved checkpoints. The best k models will be kept, others will be deleted.
_C.save_top_k = 5

# variable name of the objective
_C.objective_name = 'val_loss'

# True if we want to minimize the objective variable
_C.minimize_objective = True

# Used for k-Fold cross-validation, indicates the fold index to be used for validation.
_C.current_fold = 0

# If a path to an Ax experiment json file is given, it is used for best parameters
# and hyperparameter optimization will be skipped
_C.load_ax_experiment = False

# Number of training steps without improvement needed to trigger early stopping
_C.train_patience = False

# Number of validation steps without improvement needed to trigger early stopping
_C.val_patience = False

# Arguments for Lightning's Trainer
_C.trainer = LightningCfgNode(new_allowed=True)

# Configuration for hyperparameter optimization using Ax
_C.optimization = LightningCfgNode()

# Number of hyperparameter optimization trials
_C.optimization.total_trials = 5

# Number of folds in k-fold cross validation, 0 means disabled
_C.optimization.k_fold = 0

# List of hyperparameters according to Ax API
_C.optimization.parameters = []

# Number of training steps without improvement needed to trigger early stopping during hyperparameter optimization
_C.optimization.train_patience = False

# Number of validation steps without improvement needed to trigger early stopping during hyperparameter optimization
_C.optimization.val_patience = False

# Number of saved checkpoints during each trial. The best k models will be kept, others will be deleted.
_C.optimization.save_top_k = 1

# Arguments for Lightning's Trainer to be used for running trials
_C.optimization.trainer = LightningCfgNode(new_allowed=True)

# Hyperparameters
_C.hparams = LightningCfgNode(new_allowed=True)

# Model-specific configuration
_C.model = LightningCfgNode(new_allowed=True)

# DataModule-specific configuration
_C.data = LightningCfgNode(new_allowed=True)


def get_cfg(config_file_path):
    if config_file_path is None:
        # parse config_file input argument
        parser = ArgumentParser()
        parser.add_argument('config_file', type=FileType('r'))
        args = parser.parse_args()
        config_file_path = args.config_file.name

    cfg = _C.clone()
    cfg.merge_from_file(config_file_path)

    if cfg.source_path is False:
        cfg.source_path = os.path.dirname(config_file_path)

    cfg.source_path = os.path.abspath(cfg.source_path)
    cfg.experiment_path = timestamp_path(os.path.join(cfg.source_path, cfg.experiments_dir, cfg.experiment_name))

    cfg.freeze()
    return cfg
