import os
from abc import ABC
import copy

from ax.service.managed_loop import optimize
from ax import save
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ml_commons.ax import load_ax_experiment
from .callbacks import BatchEarlyStopping, ObjectiveMonitor
from .util import add_trainer_args, update_config_with_hparams, generate_random_hparams
from ml_commons.util.logger import get_logger

logging_logger = get_logger()


class AxLightningModule(LightningModule, ABC):
    """
    Works with pytorch-lightning v0.7.x
    """
    objective_name = 'val_loss'
    minimize_objective = True

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        self.batch_early_stop_callback = None
        self.enable_batch_early_stop = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {
            'loss': loss,
            'log': {'loss/train': loss}
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logging_logger.info(f'Epoch {self.current_epoch} val loss: {val_loss_mean:.3f}')
        return {
            'val_loss': val_loss_mean,
            'log': {'loss/val': val_loss_mean}
        }

    def on_train_start(self):
        for callback in self.trainer.callbacks:
            if isinstance(callback, BatchEarlyStopping):
                self.batch_early_stop_callback = callback
                self.enable_batch_early_stop = True
                break

    def on_batch_start(self, batch):
        # Break batch loop if batch early stop was triggered
        if self.enable_batch_early_stop and self.trainer.batch_early_stop_callback.stopped:
            return -1

    def on_save_checkpoint(self, checkpoint):
        checkpoint['config'] = self.config

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        ckpt_config = checkpoint['config']
        model = cls(ckpt_config)
        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)
        return model

    def get_current_device(self):
        return next(self.parameters()).device

    @classmethod
    def _get_early_stopping_callbacks(cls, train_patience=False, val_patience=False):
        train_loss_early_stop_callback = None
        if train_patience:
            train_loss_early_stop_callback = BatchEarlyStopping(
                monitor='loss',
                patience=train_patience,
                mode='min'
            )
        early_stop_callback = None
        if val_patience:
            early_stop_callback = EarlyStopping(
                monitor=cls.objective_name,
                patience=val_patience,
                mode='min' if cls.minimize_objective else 'max'
            )
        return train_loss_early_stop_callback, early_stop_callback

    @classmethod
    def _get_ax_evaluation_function(cls, config):
        def ax_evaluation_function(hparams):
            logging_logger.info(f'Evaluating parameters: {hparams}')

            cfg = update_config_with_hparams(config, hparams)

            # create the model
            model = cls(cfg)
            callbacks = []

            # configure early stopping
            train_loss_early_stop_patience = cfg['optimization'].get('train_patience', False)
            early_stop_patience = cfg['optimization'].get('val_patience', False)
            train_loss_early_stop_callback, early_stop_callback = cls._get_early_stopping_callbacks(
                train_loss_early_stop_patience, early_stop_patience)
            if train_loss_early_stop_callback is not None:
                callbacks.append(train_loss_early_stop_callback)

            # configure monitoring
            objective_monitor = ObjectiveMonitor(objective=cls.objective_name, minimize=True)
            callbacks.append(objective_monitor)

            # determine Trainer args
            trainer_args = add_trainer_args(
                source=cfg['trainer'],
                destination=None,
                allowed=['gradient_clip_val', 'process_position', 'num_nodes', 'gpus', 'num_tpu_cores',
                         'log_gpu_memory', 'progress_bar_refresh_rate', 'accumulate_grad_batches',
                         'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch', 'precision']
            )
            trainer_args = add_trainer_args(
                source=cfg['optimization']['trainer'],
                destination=trainer_args,
                allowed=['check_val_every_n_epoch', 'train_percent_check', 'val_percent_check', 'max_epochs',
                         'min_epochs', 'max_steps', 'min_steps', 'val_check_interval'])

            # train
            trainer = Trainer(
                logger=False,
                checkpoint_callback=False,
                early_stop_callback=early_stop_callback,
                callbacks=callbacks,
                weights_summary=None,
                num_sanity_val_steps=0,
                **trainer_args
            )
            trainer.fit(model)
            del trainer
            del model
            logging_logger.info(f'Best result: {objective_monitor.best}')
            return objective_monitor.best

        return ax_evaluation_function

    @classmethod
    def optimize_and_train(cls, config, fast_dev_run=True):
        experiment_path = config['experiment_path']
        experiment_name = config['experiment_name']

        if fast_dev_run:
            hparams = generate_random_hparams(config['optimization']['parameters'])
            cfg = update_config_with_hparams(config, hparams)
            model = cls(cfg)
            trainer = Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                weights_summary='full'
            )
            trainer.fit(model)
            del trainer
            del model

        logging_logger.info(f'Running experiment "{experiment_path}"')

        if config.get('load_ax_experiment') is not None:
            best_parameters, ax_experiment = load_ax_experiment(config['load_ax_experiment'])
        else:
            best_parameters, _, ax_experiment, _ = optimize(
                config['optimization']['parameters'],
                evaluation_function=cls._get_ax_evaluation_function(config),
                minimize=cls.minimize_objective,
                experiment_name=experiment_name,
                total_trials=config['optimization'].get('total_trials', 10)
            )
            save(ax_experiment, os.path.join(experiment_path, 'ax_experiment.json'))

        logging_logger.info(f'Training with best parameters: {best_parameters}')

        cfg = update_config_with_hparams(config, best_parameters)

        cls.fit(cfg, verbose=False)

    @classmethod
    def fit(cls, config, fast_dev_run=False, verbose=True):
        experiment_path = config['experiment_path']
        if verbose:
            logging_logger.info(f'Running experiment "{experiment_path}"')

        model = cls(config)
        callbacks = []

        tensorboard_logger = TensorBoardLogger(os.path.join(experiment_path, 'tensorboard_logs'), name='')

        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(experiment_path, 'checkpoints', f'{{epoch}}-{{{cls.objective_name}:.2f}}'),
            monitor=cls.objective_name,
            mode='min' if cls.minimize_objective else 'max',
            save_top_k=config.get('save_top_k', -1)
        )

        # configure early stopping
        train_loss_early_stop_patience = config.get('train_patience', False)
        early_stop_patience = config.get('val_patience', False)
        train_loss_early_stop_callback, early_stop_callback = cls._get_early_stopping_callbacks(
            train_loss_early_stop_patience, early_stop_patience)
        if train_loss_early_stop_callback is not None:
            callbacks.append(train_loss_early_stop_callback)

        # determine Trainer args
        trainer_args = add_trainer_args(
            source=config['trainer'],
            destination=None,
            allowed=['gradient_clip_val', 'process_position', 'num_nodes', 'gpus', 'num_tpu_cores',
                     'log_gpu_memory', 'progress_bar_refresh_rate', 'accumulate_grad_batches',
                     'distributed_backend', 'amp_level', 'reload_dataloaders_every_epoch', 'precision',
                     'check_val_every_n_epoch', 'train_percent_check', 'val_percent_check', 'max_epochs',
                     'min_epochs', 'max_steps', 'min_steps', 'val_check_interval']
        )

        if fast_dev_run:
            trainer = Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                weights_summary='full',
            )
            trainer.fit(model)
            del trainer

        trainer = Trainer(
            logger=tensorboard_logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            callbacks=callbacks,
            weights_summary='top' if (verbose and not fast_dev_run) else None,
            num_sanity_val_steps=0,
            **trainer_args
        )
        trainer.fit(model)
