from pytorch_lightning import Trainer
from yacs.config import CfgNode


class LightningCfgNode(CfgNode):
    def get_trainer_args(self):
        trainer_args = {}

        allowed_args = [
            'gradient_clip_val', 'process_position', 'num_nodes', 'num_processes', 'gpus', 'num_tpu_cores',
            'log_gpu_memory', 'progress_bar_refresh_rate', 'overfit_batches', 'track_grad_norm',
            'accumulate_grad_batches', 'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch',
            'precision', 'check_val_every_n_epoch', 'train_percent_check', 'limit_val_batches',
            'limit_test_batches', 'max_epochs', 'min_epochs', 'max_steps', 'min_steps', 'val_check_interval',
            'log_save_interval', 'row_log_interval', 'add_row_log_interval', 'print_nan_grads',
            'terminate_on_nan', 'auto_lr_find'
        ]

        depr_arg_names = Trainer.get_deprecated_arg_names()

        for arg, arg_types, arg_default in Trainer.get_init_arguments_and_types():
            if arg not in depr_arg_names and arg in allowed_args and arg in self.trainer:
                trainer_args[arg] = self.trainer[arg]

        return trainer_args

    def get_optimization_trainer_args(self):
        trainer_args = {}

        args_from_main_trainer = [
            'gradient_clip_val', 'process_position', 'num_nodes', 'gpus', 'num_tpu_cores',
            'log_gpu_memory', 'progress_bar_refresh_rate', 'accumulate_grad_batches', 'auto_lr_find',
            'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch', 'precision'
        ]

        args_from_optimization_trainer = [
            'check_val_every_n_epoch', 'train_percent_check', 'limit_val_batches', 'max_epochs',
            'min_epochs', 'max_steps', 'min_steps', 'val_check_interval', 'auto_lr_find', 'overfit_batches'
        ]

        depr_arg_names = Trainer.get_deprecated_arg_names()

        for arg, arg_types, arg_default in Trainer.get_init_arguments_and_types():
            if arg not in depr_arg_names:
                if arg in args_from_main_trainer and arg in self.trainer:
                    trainer_args[arg] = self.trainer[arg]
                if arg in args_from_optimization_trainer and arg in self.optimization.trainer:
                    trainer_args[arg] = self.optimization.trainer[arg]

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
        was_frozen = cfg.is_frozen()
        cfg.defrost()
        cfg.hparams = CfgNode(hparams)
        if was_frozen:
            cfg.freeze()
        return cfg

    def clone_with_random_hparams(self):
        hparams = self.get_random_hparams()
        return self.clone_with_hparams(hparams)

    def merge_get_cfg(self, key, cls):
        if hasattr(cls, 'get_cfg'):
            model_cfg: LightningCfgNode = cls.get_cfg()
            model_cfg.merge_from_other_cfg(getattr(self, key))
            was_frozen = self.is_frozen()
            self.defrost()
            setattr(self, key, model_cfg)
            if was_frozen:
                self.freeze()
