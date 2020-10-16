import os
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ml_commons.pytorch.lightning.util import override_lightning_logger, import_module_attr
from ml_commons.pytorch.lightning.config import get_cfg


def train():
    # parse config file with defaults
    cfg = get_cfg()

    seed_everything(cfg.seed)

    checkpoint_path = os.path.join(cfg.experiment_path, 'checkpoints', datetime.now().strftime('%y%m%d_%H%M%S'))

    override_lightning_logger()

    # instantiate model and data modules
    model_cls = import_module_attr(cfg.model_module)
    model = model_cls(cfg)
    data_cls = import_module_attr(cfg.data_module)
    data = data_cls(cfg)

    callbacks = []

    # setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, '{epoch}'),
        monitor=cfg.objective_name,
        mode='min' if cfg.minimize_objective else 'max',
        save_top_k=cfg.save_top_k,
        save_last=True
    )

    # setup early stopping
    if cfg.early_stop_patience >= 0:
        callbacks.append(EarlyStopping(
            monitor=cfg.objective_name,
            mode='min' if cfg.minimize_objective else 'max',
            patience=cfg.early_stop_patience
        ))

    # fast dev run
    fast_dev_run = cfg.trainer.pop('fast_dev_run', False)
    if fast_dev_run:
        trainer = Trainer(
            fast_dev_run=True,
            logger=False,
            checkpoint_callback=False,
            weights_summary=None
        )
        trainer.fit(model=model, datamodule=data)

    # train
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        deterministic=True,
        **cfg.trainer
    )
    trainer.fit(model=model, datamodule=data)

    # save final checkpoint
    trainer.save_checkpoint(os.path.join(checkpoint_path, 'final.ckpt'))
