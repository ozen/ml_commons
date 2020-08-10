import os
from abc import ABC
from datetime import datetime

from ax.service.managed_loop import optimize
from ax import save
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ml_commons.ax import load_ax_experiment
from ml_commons.util.logger import get_logger
from .callbacks import BatchEarlyStopping, ObjectiveMonitor
from .config import AxCfgNode

logging_logger = get_logger()


class AxLightningModule(LightningModule, ABC):
    objective_name = 'val_loss'
    minimize_objective = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        checkpoint['cfg'] = self.cfg

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        ckpt_cfg = checkpoint['cfg']
        model = cls(ckpt_cfg)
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
    def _get_ax_evaluation_function(cls, config: AxCfgNode):
        def train_and_evaluate(cfg):
            # create the model
            model = cls(cfg)
            callbacks = []

            # configure early stopping
            train_loss_early_stop_callback, early_stop_callback = cls._get_early_stopping_callbacks(
                cfg.optimization.train_patience, cfg.optimization.val_patience)
            if train_loss_early_stop_callback is not None:
                callbacks.append(train_loss_early_stop_callback)

            # configure monitoring
            objective_monitor = ObjectiveMonitor(objective=cls.objective_name, minimize=True)
            callbacks.append(objective_monitor)

            # configure model checkpointing
            if cfg.optimization.save_top_k is not None and cfg.optimization.save_top_k > 0:
                if cls.objective_name is not None:
                    checkpoint_callback = ModelCheckpoint(
                        filepath=os.path.join(cfg.experiment_path, 'checkpoints', 'trials',
                                              datetime.now().strftime('%y%m%d_%H%M%S'),
                                              f'{{epoch}}-{{{cls.objective_name}:.2f}}'),
                        monitor=cls.objective_name,
                        mode='min' if cls.minimize_objective else 'max',
                        save_top_k=cfg.optimization.save_top_k
                    )
                else:
                    checkpoint_callback = ModelCheckpoint(
                        filepath=os.path.join(cfg.experiment_path, 'checkpoints', 'trials',
                                              datetime.now().isoformat(), '{epoch}'),
                        save_top_k=cfg.optimization.save_top_k
                    )
            else:
                checkpoint_callback = None

            # train
            trainer = Trainer(
                logger=False,
                checkpoint_callback=checkpoint_callback,
                early_stop_callback=early_stop_callback,
                callbacks=callbacks,
                weights_summary=None,
                num_sanity_val_steps=0,
                **cfg.get_optimization_trainer_args()
            )
            trainer.fit(model)
            del trainer
            del model
            logging_logger.info(f'Best result: {objective_monitor.best}')
            return objective_monitor.best

        def ax_evaluation_function(hparams):
            logging_logger.info(f'Evaluating parameters: {hparams}')

            cfg = config.clone_with_hparams(hparams)

            if cfg.optimization.k_fold == 0:
                return train_and_evaluate(cfg)
            else:
                # k-fold cross validation
                logging_logger.info('k-Fold Cross Validation is enabled')
                metric = 0.0
                for fold in range(cfg.optimization.k_fold):
                    logging_logger.info(f'Fold {fold+1}/{cfg.optimization.k_fold}')
                    cfg.defrost()
                    cfg.current_fold = fold
                    cfg.freeze()
                    metric += train_and_evaluate(cfg)
                metric = metric / cfg.optimization.k_fold
                logging_logger.info(f'CV average of best results: {metric}')
                return metric

        return ax_evaluation_function

    @classmethod
    def optimize_and_train(cls, cfg: AxCfgNode, fast_dev_run=True):
        """
        Runs Ax optimization loop
        """
        assert isinstance(cfg, AxCfgNode)
        assert cls.objective_name is not None

        if fast_dev_run:
            cfg = cfg.clone_with_random_hparams()
            model = cls(cfg)
            trainer = Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                auto_lr_find=False,
                weights_summary='full'
            )
            trainer.fit(model)
            del trainer
            del model

        logging_logger.info(f'Running experiment "{cfg.experiment_path}"')

        if cfg.load_ax_experiment:
            best_parameters, ax_experiment = load_ax_experiment(cfg.load_ax_experiment)
        else:
            best_parameters, _, ax_experiment, _ = optimize(
                cfg.optimization.parameters,
                evaluation_function=cls._get_ax_evaluation_function(cfg),
                minimize=cls.minimize_objective,
                experiment_name=cfg.experiment_name,
                total_trials=cfg.optimization.total_trials
            )
            save(ax_experiment, os.path.join(cfg.experiment_path, 'ax_experiment.json'))

        logging_logger.info(f'Training with best parameters: {best_parameters}')

        cfg = cfg.clone_with_hparams(best_parameters)
        cls.fit(cfg, verbose=False)

    @classmethod
    def fit(cls, cfg: AxCfgNode, fast_dev_run=False, verbose=True):
        """
        Runs a single train loop
        """
        assert isinstance(cfg, AxCfgNode)

        if verbose:
            logging_logger.info(f'Running experiment "{cfg.experiment_path}"')

        model = cls(cfg)
        callbacks = []

        tensorboard_logger = TensorBoardLogger(os.path.join(cfg.experiment_path, 'tensorboard_logs'), name='')

        if cls.objective_name is not None:
            checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(cfg.experiment_path, 'checkpoints', f'{{epoch}}-{{{cls.objective_name}:.2f}}'),
                monitor=cls.objective_name,
                mode='min' if cls.minimize_objective else 'max',
                save_top_k=cfg.save_top_k
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(cfg.experiment_path, 'checkpoints', '{epoch}'),
                save_top_k=-1
            )

        # configure early stopping
        train_loss_early_stop_callback, early_stop_callback = cls._get_early_stopping_callbacks(
            cfg.train_patience, cfg.val_patience)
        if train_loss_early_stop_callback is not None:
            callbacks.append(train_loss_early_stop_callback)

        if fast_dev_run:
            trainer_args = cfg.get_trainer_args()
            del trainer_args['auto_lr_find']
            trainer = Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                auto_lr_find=False,
                weights_summary='full',
                **trainer_args
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
            **cfg.get_trainer_args()
        )
        trainer.fit(model)
