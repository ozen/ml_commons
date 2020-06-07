import os

from pytorch_lightning import Trainer
from yacs.config import CfgNode

from ml_commons.util.path import uniquify_path


class AxCfgNode(CfgNode):
    def get_optimization_trainer_args(self):
        trainer_args = {}

        args_from_main_trainer = [
            'gradient_clip_val', 'process_position', 'num_nodes', 'gpus', 'num_tpu_cores',
            'log_gpu_memory', 'progress_bar_refresh_rate', 'accumulate_grad_batches',
            'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch', 'precision'
        ]

        args_from_optimization_trainer = [
            'check_val_every_n_epoch', 'train_percent_check', 'val_percent_check', 'max_epochs',
            'min_epochs', 'max_steps', 'min_steps', 'val_check_interval'
        ]

        depr_arg_names = Trainer.get_deprecated_arg_names()

        for arg, arg_types, arg_default in Trainer.get_init_arguments_and_types():
            if arg not in depr_arg_names:
                if arg in args_from_main_trainer:
                    trainer_args[arg] = self.trainer[arg]
                elif arg in args_from_optimization_trainer:
                    trainer_args[arg] = self.optimization.trainer[arg]

        return trainer_args

    def get_trainer_args(self):
        trainer_args = {}

        allowed_args = [
            'gradient_clip_val', 'process_position', 'num_nodes', 'gpus', 'num_tpu_cores',
            'log_gpu_memory', 'progress_bar_refresh_rate', 'accumulate_grad_batches',
            'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch', 'precision',
            'check_val_every_n_epoch', 'train_percent_check', 'val_percent_check', 'max_epochs',
            'min_epochs', 'max_steps', 'min_steps', 'val_check_interval'
        ]

        depr_arg_names = Trainer.get_deprecated_arg_names()

        for arg, arg_types, arg_default in Trainer.get_init_arguments_and_types():
            if arg not in depr_arg_names and arg in allowed_args and arg in self.trainer:
                trainer_args[arg] = self.trainer[arg]

        return trainer_args

    def get_random_hparams(self):
        hparams = {}
        for param in self.optimization.parameters:
            if param['type'] == 'range':
                hparams[param['name']] = param['bounds'][0]
            elif param['type'] == 'choice':
                hparams[param['name']] = param['values'][0]
        return hparams

    def clone_with_hparams(self, hparams):
        cfg = self.clone()
        cfg.hparams = CfgNode(hparams)
        return cfg

    def clone_with_random_hparams(self):
        hparams = self.get_random_hparams()
        return self.clone_with_hparams(hparams)


_C = AxCfgNode()

# The root path. Defaults to the directory of config file.
_C.source_path = False
# Module path to the AxLightningModule, relative to source path
_C.module = 'module.AxLightningModule'
# Path to the experiments directory, relative to source path
_C.experiments_dir = 'experiments'
# Name of the experiment
_C.experiment_name = 'default'
# filename patterns relative to the module's directory to be saved to the logs. beware of loops.
_C.log_files = ['*']
# Path to the data directory
_C.data_root = 'data'

# Batch size
_C.batch_size = 256
#
_C.save_top_k = 5
#
_C.load_ax_experiment = False
#
_C.train_patience = False

_C.val_patience = False

_C.trainer = AxCfgNode(new_allowed=True)

_C.optimization = AxCfgNode()

_C.optimization.total_trials = 5

_C.optimization.parameters = []

_C.optimization.train_patience = False

_C.optimization.val_patience = False

_C.optimization.trainer = AxCfgNode(new_allowed=True)

_C.hparams = AxCfgNode(new_allowed=True)

_C.model = AxCfgNode(new_allowed=True)


def get_cfg(config_file_path):
    cfg = _C.clone()
    cfg.merge_from_file(config_file_path)

    if cfg.source_path is False:
        cfg.source_path = os.path.dirname(config_file_path)

    cfg.source_path = os.path.abspath(cfg.source_path)
    cfg.experiment_path = uniquify_path(os.path.join(cfg.source_path, cfg.experiments_dir, cfg.experiment_name))

    cfg.freeze()
    return cfg
