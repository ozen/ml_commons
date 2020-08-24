import os
from datetime import datetime

import pytorch_lightning as pl
import ax
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ml_commons.ax import load_ax_experiment
from ml_commons.pytorch.lightning.callbacks import BatchEarlyStopping, ObjectiveMonitor
from ml_commons.pytorch.lightning.main import setup_experiment
from ml_commons.util.logger import get_logger

logging_logger = get_logger()


class Experimenter:
    def __init__(self, config_file=None):
        model_class, data_class, cfg = setup_experiment(config_file)
        self.model_class = model_class
        self.data_class = data_class
        self.cfg = cfg

    def _get_early_stopping_callbacks(self, train_patience=False, val_patience=False):
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
                monitor=self.cfg.objective_name,
                patience=val_patience,
                mode='min' if self.cfg.minimize_objective else 'max'
            )
        return train_loss_early_stop_callback, early_stop_callback

    def _get_checkpoint_callback(self, filepath, save_top_k):
        if self.cfg.objective_name is not None:
            checkpoint_callback = ModelCheckpoint(
                filepath=filepath,
                monitor=self.cfg.objective_name,
                mode='min' if self.cfg.minimize_objective else 'max',
                save_top_k=save_top_k
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                filepath=filepath,
                save_top_k=-1
            )
        return checkpoint_callback

    def _get_ax_evaluation_function(self):
        def train_and_evaluate(cfg):
            # create the model
            model = self.model_class(cfg)
            callbacks = []

            # configure early stopping
            train_loss_early_stop_callback, early_stop_callback = self._get_early_stopping_callbacks(
                cfg.optimization.train_patience, cfg.optimization.val_patience)
            if train_loss_early_stop_callback is not None:
                callbacks.append(train_loss_early_stop_callback)

            # configure monitoring
            objective_monitor = ObjectiveMonitor(objective=self.cfg.objective_name,
                                                 minimize=self.cfg.minimize_objective)
            callbacks.append(objective_monitor)

            # configure model checkpointing
            if cfg.optimization.save_top_k is not None and cfg.optimization.save_top_k > 0:
                checkpoint_callback = self._get_checkpoint_callback(
                    filepath=os.path.join(cfg.experiment_path, 'checkpoints', 'trials',
                                          datetime.now().strftime('%y%m%d_%H%M%S'),
                                          f'{{epoch}}-{{{self.cfg.objective_name}:.2f}}'),
                    save_top_k=cfg.optimization.save_top_k
                )
            else:
                checkpoint_callback = None

            # train
            trainer = pl.Trainer(
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

            cfg = self.cfg.clone_with_hparams(hparams)

            if cfg.optimization.k_fold == 0:
                return train_and_evaluate(cfg)
            else:
                # k-fold cross validation
                logging_logger.info('k-Fold Cross Validation is enabled')
                metric = 0.0
                for fold in range(cfg.optimization.k_fold):
                    logging_logger.info(f'Fold {fold + 1}/{cfg.optimization.k_fold}')
                    cfg.defrost()
                    cfg.current_fold = fold
                    cfg.freeze()
                    metric += train_and_evaluate(cfg)
                metric = metric / cfg.optimization.k_fold
                logging_logger.info(f'CV average of best results: {metric}')
                return metric

        return ax_evaluation_function

    def optimize_and_fit(self, fast_dev_run=True):
        """
        Runs Ax optimization loop
        """
        data = self.data_class(self.cfg)

        if fast_dev_run:
            cfg = self.cfg.clone_with_random_hparams()
            model = self.model_class(cfg)
            trainer = pl.Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                auto_lr_find=False,
                weights_summary='full'
            )
            trainer.fit(model, data)
            del trainer
            del model

        logging_logger.info(f'Running experiment "{self.cfg.experiment_path}"')

        if self.cfg.load_ax_experiment:
            best_parameters, ax_experiment = load_ax_experiment(self.cfg.load_ax_experiment)
        else:
            best_parameters, _, ax_experiment, _ = ax.optimize(
                self.cfg.optimization.parameters,
                evaluation_function=self._get_ax_evaluation_function(),
                minimize=self.cfg.minimize_objective,
                experiment_name=self.cfg.experiment_name,
                total_trials=self.cfg.optimization.total_trials
            )
            ax.save(ax_experiment, os.path.join(self.cfg.experiment_path, 'ax_experiment.json'))

        logging_logger.info(f'Training with best parameters: {best_parameters}')

        self.cfg = self.cfg.clone_with_hparams(best_parameters)
        self.fit(verbose=False)

    def fit(self, fast_dev_run=False, verbose=True):
        """
        Runs a single train loop
        """
        if verbose:
            logging_logger.info(f'Running experiment "{self.cfg.experiment_path}"')

        model = self.model_class(self.cfg)
        data = self.data_class(self.cfg)
        callbacks = []

        # dump the last version of the config
        with open(os.path.join(self.cfg.experiment_path, 'final_config.yaml'), 'w') as f:
            self.cfg.dump(stream=f)

        tensorboard_logger = TensorBoardLogger(os.path.join(self.cfg.experiment_path, 'tensorboard_logs'), name='')

        # configure model checkpointing
        checkpoint_callback = self._get_checkpoint_callback(
            filepath=os.path.join(self.cfg.experiment_path, 'checkpoints',
                                  f'{{epoch}}-{{{self.cfg.objective_name}:.2f}}'),
            save_top_k=self.cfg.save_top_k
        )

        # configure early stopping
        train_loss_early_stop_callback, early_stop_callback = self._get_early_stopping_callbacks(
            self.cfg.train_patience, self.cfg.val_patience)
        if train_loss_early_stop_callback is not None:
            callbacks.append(train_loss_early_stop_callback)

        if fast_dev_run:
            trainer_args = self.cfg.get_trainer_args()
            del trainer_args['auto_lr_find']
            trainer = pl.Trainer(
                fast_dev_run=True,
                logger=False,
                checkpoint_callback=False,
                auto_lr_find=False,
                weights_summary='full',
                **trainer_args
            )
            trainer.fit(model, data)
            del trainer

        trainer = pl.Trainer(
            logger=tensorboard_logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            callbacks=callbacks,
            weights_summary='top' if (verbose and not fast_dev_run) else None,
            num_sanity_val_steps=0,
            **self.cfg.get_trainer_args()
        )
        trainer.fit(model, data)
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        trainer.save_checkpoint(os.path.join(self.cfg.experiment_path, 'checkpoints', f'final_{timestamp}.ckpt'))

    def resume(self, checkpoint_path, override_cfg=None):
        model = self.model_class.load_from_checkpoint(checkpoint_path)
        cfg = model.cfg
        if override_cfg is not None:
            cfg.merge_from_list(override_cfg)

        data = self.data_class(cfg)

        callbacks = []

        tensorboard_logger = TensorBoardLogger(os.path.join(cfg.experiment_path, 'tensorboard_logs'), name='')

        # configure model checkpointing
        checkpoint_callback = self._get_checkpoint_callback(
            filepath=os.path.join(cfg.experiment_path, 'checkpoints', f'{{epoch}}-{{{cfg.objective_name}:.2f}}'),
            save_top_k=cfg.save_top_k
        )

        # configure early stopping
        train_loss_early_stop_callback, early_stop_callback = self._get_early_stopping_callbacks(
            cfg.train_patience, cfg.val_patience)
        if train_loss_early_stop_callback is not None:
            callbacks.append(train_loss_early_stop_callback)

        trainer = pl.Trainer(
            resume_from_checkpoint=checkpoint_path,
            logger=tensorboard_logger,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            callbacks=callbacks,
            **cfg.get_trainer_args()
        )
        trainer.fit(model, data)
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        trainer.save_checkpoint(os.path.join(self.cfg.experiment_path, 'checkpoints', f'final_{timestamp}.ckpt'))
