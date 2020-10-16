from argparse import ArgumentParser, FileType

from yacs.config import CfgNode


_C = CfgNode()

_C.experiment_path = ''

_C.experiment_name = ''

_C.model_module = ''

_C.data_module = ''

_C.seed = 123

_C.objective_name = 'val_loss'

_C.minimize_objective = True

_C.batch_size = 64

_C.num_workers = 0

_C.save_top_k = -1

_C.early_stop_patience = -1     # -1 to disable

_C.model = CfgNode(new_allowed=True)

_C.data = CfgNode(new_allowed=True)

_C.trainer = CfgNode(new_allowed=True)

_C.tune = CfgNode()

_C.hparams = CfgNode(new_allowed=True)


def get_cfg(config_file_path=None):
    if config_file_path is None:
        parser = ArgumentParser()
        parser.add_argument('config_file', type=FileType('r'), default='config.yaml')
        args = parser.parse_args()
        config_file_path = args.config_file.name

    cfg = _C.clone()
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg
