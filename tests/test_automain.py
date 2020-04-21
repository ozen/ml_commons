import sys
import os
from ml_commons.pytorch.lightning import automain

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@automain
def main(cls, config):
    cls.optimize_and_train(config)
