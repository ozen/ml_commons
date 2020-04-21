from ml_commons.pytorch.lightning import automain


@automain
def main(cls, config):
    cls.optimize_and_train(config)
