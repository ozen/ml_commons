import logging
import numpy as np
from torch import is_tensor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.base import Callback

logging_logger = logging.getLogger(__name__)


class BatchEarlyStopping(Callback):
    def __init__(self, monitor: str = 'loss', min_delta: float = 0.0, patience: int = 0, mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.early_stopping = None
        self.iteration = 0
        self.stopped = False

    def on_train_start(self, trainer, pl_module):
        # Allow instances to be re-used
        self.early_stopping = EarlyStopping(monitor=self.monitor, min_delta=self.min_delta,
                                            patience=self.patience, mode=self.mode, verbose=False)
        self.iteration = 0
        self.stopped = False

    def on_batch_end(self, trainer, pl_module):
        self.iteration += 1
        self.stopped = self.early_stopping.on_epoch_end(trainer, pl_module)
        return self.stopped

    def on_train_end(self, trainer, pl_module):
        if self.iteration > 0:
            logging_logger.info(f'Iteration {self.iteration + 1:05d}: early stopping')


class ObjectiveMonitor(Callback):
    def __init__(self, objective, minimize=True):
        self.objective = objective
        self.op = np.less if minimize else np.greater
        self.best = np.Inf if self.op == np.less else -np.Inf

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.objective)
        if is_tensor(current):
            current = current.cpu().detach().item()
        if self.op(current, self.best):
            self.best = current
