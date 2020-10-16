import os
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune as ray_tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.logger import MLFLowLogger
from yacs.config import CfgNode

from ml_commons.pytorch.lightning.config import get_cfg
from ml_commons.pytorch.lightning.util import override_lightning_logger, import_module_attr


def tune():
    def trainable(hparams):
        seed_everything(cfg.seed)

        # add hparams to cfg
        cfg.defrost()
        cfg.hparams = CfgNode(hparams)
        cfg.freeze()

        checkpoint_path = os.path.join(cfg.experiment_path, 'checkpoints', datetime.now().strftime('%y%m%d_%H%M%S'))

        # instantiate model and data modules
        model_cls = import_module_attr(cfg.model_module)
        model = model_cls(cfg)
        data_cls = import_module_attr(cfg.data_module)
        data = data_cls(cfg)

        # setup model checkpointing
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, '{epoch}'),
            monitor=cfg.objective_name,
            mode='min' if cfg.minimize_objective else 'max',
            save_top_k=cfg.save_top_k,
            save_last=True
        )

        tune_callback = TuneReportCheckpointCallback(metrics=cfg.tune.metrics)

        # train
        cfg.trainer.pop('progress_bar_refresh_rate', 0)
        trainer = Trainer(
            checkpoint_callback=checkpoint_callback,
            callbacks=[tune_callback],
            deterministic=True,
            progress_bar_refresh_rate=0,
            **cfg.trainer
        )
        trainer.fit(model=model, datamodule=data)

        # save final checkpoint
        trainer.save_checkpoint(os.path.join(checkpoint_path, 'final.ckpt'))

    # parse config file with defaults
    cfg = get_cfg()

    override_lightning_logger()

    # fast dev run
    fast_dev_run = cfg.trainer.pop('fast_dev_run', False)
    if fast_dev_run:
        model_cls = import_module_attr(cfg.model_module)
        model = model_cls(cfg)
        data_cls = import_module_attr(cfg.data_module)
        data = data_cls(cfg)
        trainer = Trainer(
            fast_dev_run=True,
            logger=False,
            checkpoint_callback=False,
            weights_summary=None
        )
        trainer.fit(model=model, datamodule=data)

    search_space = import_module_attr(cfg.tune.search_space)

    loggers = []
    # TODO MLFLowLogger

    reporter = CLIReporter()

    searcher = None
    if isinstance(cfg.tune.searcher, str) and len(cfg.tune.searcher) > 0:
        searcher = ray_tune.create_searcher(search_alg=cfg.tune.searcher, **cfg.tune.searcher_parameters)

    scheduler = None
    if isinstance(cfg.tune.scheduler, str) and len(cfg.tune.scheduler) > 0:
        scheduler = ray_tune.create_scheduler(scheduler=cfg.tune.scheduler, **cfg.tune.scheduler_parameters)

    ray_tune.run(
        run_or_experiment=trainable,
        name=cfg.experiment_name,
        metric=cfg.objective_name,
        mode='min' if cfg.minimize_objective else 'max',
        config=search_space,
        resources_per_trial=cfg.tune.resources_per_trial,
        num_samples=cfg.tune.num_samples,
        local_dir=os.path.join(cfg.experiment_path, 'ray_results'),
        search_alg=searcher,
        scheduler=scheduler,
        progress_reporter=reporter,
        loggers=loggers,
        log_to_file=True
    )