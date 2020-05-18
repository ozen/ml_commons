from ml_commons.pytorch.lightning import automain


@automain
def main(cls, cfg):
    cls.optimize_and_train(cfg)
